import argparse

import sys
import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import operator
from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.modules.linear_growing_module import LinearGrowingModule, LinearMergeGrowingModule
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import re

if '/home/tau/sdouka/codebase/experimental_grow' not in sys.path:
    sys.path.append('/home/tau/sdouka/codebase/experimental_grow')
if "/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/" not in sys.path:
    sys.path.append("/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/")

from tools.gpu_tracking import GpuTracker
from tools.logger import Logger
from tools.datasets import get_dataset
from tools.utils import evaluate_dataset, mini_batch_gradient_descent, line_search, DAG_to_pyvis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="Debug")
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--node_name", type=str)
    parser.add_argument("--iters", type=int, default=3)
    # parser.add_argument(
    #     "--parallel_edges", action=argparse.BooleanOptionalAction, default=True
    # )
    parser.add_argument("--neurons", type=int, default=20)
    parser.add_argument("--neuron_epochs", type=int, default=100)
    parser.add_argument("--neuron_lrate", type=float, default=1e-3)
    parser.add_argument("--neuron_batch_size", type=int, default=256)
    # parser.add_argument("--new_opt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--inter_train", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument(
        "--random_growth", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--restrict_actions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--estimate_dependencies", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    return args


def setup_experiment_tags():
    repo = git.Repo(search_parent_directories=True) # type: ignore
    git_commit = repo.head.object.hexsha
    try:
        gpu_index = torch.cuda.current_device()
    except:
        gpu_index = None
    tags = {
        "git.commit": git_commit,
        "slurm.job_id": args.job_id,
        "slurm.node_name": args.node_name,
        "gpu_index": gpu_index,
    }
    return tags

def eval_fn() -> None:
    global global_epoch
    val_acc, val_loss = evaluate_dataset(
        model,
        val_dataloader,
        loss_fn=model.growing_dag.loss_fn
    )
    logger.log_metric(
        "Intermediate training/val loss",
        val_loss,
        global_epoch,
    )
    logger.log_metric(
        "Intermediate training/val accuracy",
        val_acc,
        global_epoch,
    )
    test_acc, test_loss = evaluate_dataset(
        model,
        test_dataloader,
        loss_fn=model.growing_dag.loss_fn
    )
    logger.log_metric(
        "Intermediate training/test loss",
        test_loss,
        global_epoch,
    )
    logger.log_metric(
        "Intermediate training/test accuracy",
        test_acc,
        global_epoch,
    )
    global_epoch += 1

# acc_test = []
# indices = []

def step(model, 
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    val_dataloader : DataLoader,
    loss_fn,
    logger: Logger,
):
    # Find new ways to grow the DAG
    generations = model.growing_dag.dag.define_next_actions()
    logger.log_metric("complexity/nb of actions", len(generations), global_step)

    # Initialize activities
    pre_activities_grad = {
        node: torch.empty(0) for node in model.growing_dag.dag.nodes if node!=model.growing_dag.dag.root
    }
    input_B = {node: torch.empty(0) for node in model.growing_dag.dag.nodes}
    bottleneck = {}

    # Initialize tensors
    model.init_computation()
    # Forward - backward loop
    for X, Y in train_dataloader:
        X, Y = X.to(model.device), Y.to(model.device)
        model.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y) # TODO: check loss reduction
        loss.backward()
        # Update tensors
        model.update_computation()

        # Accumulate pre-activity gradients and input tensors on cpu
        for node_module in model.growing_dag.dag.get_all_node_modules():
            assert node_module.activity is not None
            # Save input activity of input layers
            input_B[node_module._name] = torch.cat(
                (
                    input_B[node_module._name],
                    node_module.activity.clone().detach().cpu(),
                )
            )
            if node_module._name == model.growing_dag.dag.root:
                continue
            assert node_module.pre_activity is not None
            assert node_module.pre_activity.grad is not None
            # Save pre activiy gradients
            pre_activities_grad[node_module._name] = torch.cat(
                (
                    pre_activities_grad[node_module._name],
                    node_module.pre_activity.grad.clone().detach().cpu(),
                )
            )
    
    # Compute optimal updates
    model.compute_optimal_delta()

    # Retrieve expressivity bottleneck and inputs on important nodes
    with torch.no_grad():
        for node_module in model.growing_dag.dag.get_all_node_modules():
            if node_module._name == model.growing_dag.dag.root:
                continue
            # Compute expressivity bottleneck
            v_proj = pre_activities_grad[node_module._name]
            for module in node_module.previous_modules:
                v_proj -= (
                    module.optimal_delta_layer(
                        input_B[module.previous_module._name].to(module.device)
                    ).cpu()
                )

            bottleneck[node_module._name] = v_proj

    del pre_activities_grad
    
    bott_norms = {key: torch.linalg.norm(val) for key, val in bottleneck.items()}
    important_node = max(bott_norms.items(), key=operator.itemgetter(1))[0]
    print(
        f"Restricting action space to node {important_node} with norm {bott_norms[important_node]}"
    )
    generations = model.growing_dag.restrict_action_space(generations, important_node)
    # generations = model.growing_dag.restrict_action_space(generations, chosen_outputs=[important_node])

    # if estimate_dependencies and len(generations) > 3:
    #     input_B = {name:value for name, value in input_B.items() if name not in nx.descendants(model.growing_dag.dag, important_node) and name != important_node}
    #     hsic = calculate_dependency(input_B, bottleneck[important_node], n_samples=1000)
    #     hsic_values = torch.tensor(list(hsic.values()), device=global_device(), dtype=float)
    #     percentile = torch.quantile(hsic_values, 0.9)
    #     dominant_inputs = [name for name, value in hsic.items() if value >= percentile]
    #     generations = self.restrict_action_space(generations, chosen_inputs=dominant_inputs)
    #     for name, value in hsic.items():
    #         logger.log_metric(f"actions/HSIC/node {name}", value, global_step)
    #     logger.log_metric("actions/HSIC 0.9 quantile", percentile, global_step)
    
    logger.log_metric("complexity/nb of tested actions", len(generations), global_step)
    
    # Reset all hooks
    model.reset_computation()
    
    # Execute all graph growth actions
    model.growing_dag.execute_expansions(
        actions=generations,
        bottleneck=bottleneck,
        input_B=input_B,
        amplitude_factor=False,
        evaluate=False,
        verbose=False,
    )

    # Compute amplitude factor
    for expansion in generations:
        mask = {
            "nodes": [expansion.expanding_node],
            "edges": expansion.new_edges,
        }
        def simulate_loss(factor):
            model.set_scaling_factor(factor)

            loss = []
            with torch.no_grad():
                for x, y in dev_dataloader:
                    x = x.to(model.device)
                    y = y.to(model.device)
                    pred, _ = model.extended_forward(x, mask=mask)
                    loss.append(loss_fn(pred, y).item())

            return np.mean(loss).item()

        factor, _ = line_search(simulate_loss, verbose=False)
        model.set_scaling_factor(factor)
        expansion.metrics["scaling_factor"] = factor
        expansion.evaluate(model, train_dataloader, dev_dataloader, val_dataloader, loss_fn=loss_fn)

    # Find action that generates minimum loss
    model.growing_dag.choose_growth_best_action(
        generations, use_bic=False, verbose=True
    )
    factor = model.growing_dag.chosen_action.metrics["scaling_factor"]
    logger.log_metric("growth/amplitude factor", factor, global_step)

    # Apply growth action
    model.growing_dag.apply_change()
    
    # Delete all updates
    model.growing_dag.delete_update()

    # Update sizes
    model.update_size()

    # Memory optimization
    gc.collect()
    torch.cuda.empty_cache()

    return model

def grow_network(
    model: GrowingContainer,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    val_dataloader: DataLoader,
    inter_train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    steps: int,
    train_epochs: int,
    loss_fn,
    # inter_train: bool,
    # random_growth: bool,
    # restrict_actions: bool,
    # estimate_dependencies: bool,
    logger: Logger,
    verbose: bool = False,
):
    global global_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_history, acc_history = mini_batch_gradient_descent(
        model=model,
        cost_fn=loss_fn,
        dataloader=inter_train_dataloader,
        optimizer=optimizer,
        max_epochs=train_epochs,
        fast=False,
        eval_fn=eval_fn,
        verbose=False,
    )
    for i in range(len(loss_history)):
        logger.log_metric(
            "Intermediate training/dev loss",
            loss_history[i],
            global_epoch - train_epochs + i,
        )
        logger.log_metric(
            "Intermediate training/dev accuracy",
            acc_history[i],
            global_epoch - train_epochs + i,
        )


    global global_step
    for global_step in tqdm(range(steps)):
        print("\nStep", global_step + 1)
        
        # with GpuTracker(gpu_index=[tags["gpu_index"]], logger=logger) as tracker:
        step(
            model,
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            logger=logger,
        )

        # Evaluation after growth
        growth_acc_train, growth_loss_train = evaluate_dataset(model, train_dataloader, loss_fn=loss_fn)
        growth_acc_dev, growth_loss_dev = evaluate_dataset(model, dev_dataloader, loss_fn=loss_fn)
        growth_acc_val, growth_loss_val = evaluate_dataset(model, val_dataloader, loss_fn=loss_fn)
        growth_acc_test, growth_loss_test = evaluate_dataset(model, test_dataloader, loss_fn=loss_fn)

        # Intermediate training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        hist_loss_dev, hist_acc_dev = mini_batch_gradient_descent(
            model=model,
            cost_fn=loss_fn,
            dataloader=inter_train_dataloader,
            optimizer=optimizer,
            max_epochs=train_epochs,
            fast=False,
            eval_fn=eval_fn,
            verbose=False,
        )
        for i in range(len(loss_history)):
            logger.log_metric(
                "Intermediate training/dev loss",
                loss_history[i],
                global_epoch - train_epochs + i,
            )
            logger.log_metric(
                "Intermediate training/dev accuracy",
                acc_history[i],
                global_epoch - train_epochs + i,
            )

        # Evaluation
        acc_dev, loss_dev = evaluate_dataset(model, dev_dataloader, loss_fn)
        acc_val, loss_val = evaluate_dataset(model, val_dataloader, loss_fn)
        acc_test, loss_test = evaluate_dataset(model, test_dataloader, loss_fn)
        print(f"acc_dev={hist_acc_dev[-1]} loss_dev={hist_loss_dev[-1]}")
        print(f"acc_val={acc_val} loss_val={loss_val}")
        print(f"acc_test{acc_test} loss_test={loss_test}")
        print(f"growth_acc_test={growth_acc_test} growth_loss_test={growth_loss_test}")

        # logger.log_all_metrics(step=global_step)

        logger.log_metric("growth train loss", growth_loss_train, global_epoch)
        logger.log_metric("growth dev loss", growth_loss_dev, global_epoch)
        logger.log_metric("dev loss", loss_dev, global_epoch)
        logger.log_metric("growth val loss", growth_loss_val, global_epoch)
        logger.log_metric("val loss", loss_val, global_epoch)
        logger.log_metric("growth test loss", growth_loss_test, global_epoch)
        logger.log_metric("test loss", loss_test, global_epoch)
        
        logger.log_metric("growth train accuracy", growth_acc_train, global_epoch)
        logger.log_metric("growth dev accuracy", growth_acc_dev, global_epoch)
        logger.log_metric("dev accuracy", acc_dev, global_epoch)
        logger.log_metric("growth val accuracy", growth_acc_val, global_epoch)
        logger.log_metric("val accuracy", acc_val, global_epoch)
        logger.log_metric("growth test accuracy", growth_acc_test, global_epoch)
        logger.log_metric("test accuracy", acc_test, global_epoch)
        logger.log_metric(
            "complexity/nb of parameters",
            model.growing_dag.dag.count_parameters_all(), # type: ignore
            global_epoch,
        )
        # nb of parameters per edge
        for edge in model.growing_dag.dag.edges: # type: ignore
            edge0 = re.sub("@", ".", edge[0])
            edge1 = re.sub("@", ".", edge[1])
            params = model.growing_dag.dag.count_parameters([edge]) # type: ignore
            logger.log_metric(
                f"complexity/nb of parameters at/layer {edge0}_{edge1}",
                params,
                global_epoch,
            )
        # in-degree and out-degree per node
        for node in model.growing_dag.dag.nodes: # type: ignore
            _node = re.sub("@", ".", node)
            # logger.log_metric(
            #     f"complexity/in-degree/node {_node}",
            #     model.growing_dag.dag.in_degree(node), # type: ignore
            #     global_epoch,
            # )
            # logger.log_metric(
            #     f"complexity/out-degree/node {_node}",
            #     model.growing_dag.dag.out_degree(node), # type: ignore
            #     global_epoch,
            # )
            logger.log_metric(
                f"complexity/size/node {_node}",
                model.growing_dag.dag.nodes[node]["size"], # type: ignore
                global_epoch,
            )

        with torch.no_grad():
            # Save model
            try:
                # TODO: save dag instead
                logger.log_pytorch_model(
                    model, f"Growable CNN step {global_step}", train_dataloader.dataset.dataset.data
                )
            except Exception as error:
                print(f"[DAGNN Model] {error}")

        # Save interactive graph
        dirname = "temp"
        try:
            graph = DAG_to_pyvis(model.growing_dag.dag) # type: ignore
            pyvis_path = f"{dirname}/graph_.html"
            graph.save_graph(pyvis_path)
            logger.log_artifact(pyvis_path)
        except Exception as error:
            print(f"[Interactive DAG] {error}")


        # Memory optimization
        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            print("\n********* NEW GRAPH *********")
            model.growing_dag.dag.draw() # type: ignore

    return model

class CNN(GrowingContainer):
    def __init__(self, net, out_features, device) -> None:
        input_volume = net.dag.get_node_module(net.dag.root).input_volume
        output_volume = net.dag.get_node_module(net.dag.end).output_volume
        super().__init__(in_features=input_volume, out_features=out_features, device=device)
        self.growing_dag = net

        self.linear_merge = LinearMergeGrowingModule(
            in_features=output_volume,
            previous_modules=[self.growing_dag.dag.get_node_module(self.growing_dag.dag.end)],
            name="linear_merge",
        ).to(device)
        self.mlp1 = LinearGrowingModule(
            in_features=output_volume,
            out_features=50,
            post_layer_function=torch.nn.SELU(),
            previous_module=self.linear_merge,
            name="mlp1",
            device=device,
        )
        self.mlp2 = torch.nn.Linear(
            in_features=50,
            out_features=out_features,
        ).to(device)

        self.growing_dag.dag.get_node_module(self.growing_dag.dag.end).set_next_modules([self.linear_merge])
        self.growing_dag.dag.get_node_module(self.growing_dag.dag.end).post_merge_function = torch.nn.Sequential(
            torch.nn.SELU(), # NOTE: SHOULD BE HERE!!
        )
        self.linear_merge.set_next_modules([self.mlp1])
        self.mlp1.next_module = self.mlp2

        self.set_growing_layers()
    
    def set_growing_layers(self):
        self._growing_layers.append(self.growing_dag)
    
    def forward(self, x):
        x = self.growing_dag(x)
        x = torch.nn.Flatten()(x)
        x = self.linear_merge(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x
    
    def extended_forward(self, x, mask={}):
        x, _ = self.growing_dag.extended_forward(x, mask=mask)
        x = torch.nn.Flatten()(x)
        x = self.linear_merge(x)
        x = self.mlp1(x)
        return self.mlp2(x)


if __name__ == "__main__":
    args = parse_args()

    global_epoch = 0

    trainset, valset, testset = get_dataset("cifar10", "data", split_train_val=0.4)
    valsize = int(0.5 * len(valset)) # type: ignore
    devset, valset = torch.utils.data.random_split(valset, [valsize, valsize])
    print(f"{len(trainset)=}, {len(devset)=}, {len(valset)=}, {len(testset)=}") # type: ignore
    print(f"{trainset.dataset.data.shape=}") # type: ignore
    in_channels = trainset.dataset.data.shape[-1] # type: ignore
    out_features = len(np.unique(trainset.dataset.targets)) # type: ignore
    input_shape = trainset.dataset.data.shape[1:-1] # type: ignore
    print(f"{in_channels=} {out_features=} {input_shape=}")

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(devset, batch_size=512)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=1024)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1024)

    indices = train_dataloader.dataset.indices + dev_dataloader.dataset.indices # type: ignore
    inter_train_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataloader.dataset.dataset, indices), batch_size=512, shuffle=True) # type: ignore

    dag_out_channels = 50
    net = GrowingGraphNetwork(
        in_features=in_channels,
        out_features=dag_out_channels,
        neurons=args.neurons,
        neuron_epochs=args.neuron_epochs,
        neuron_lrate=args.neuron_lrate,
        neuron_batch_size=args.neuron_batch_size,
        loss_fn=torch.nn.CrossEntropyLoss(),
        layer_type="convolution",
        input_shape=input_shape,
    )

    print(net.dag)
    # net.dag.draw()
    print(net.device)
    print()

    model = CNN(net, out_features=out_features, device=net.device)

    # data_rng = torch.Generator()
    logger = Logger(experiment_name=args.exp_name, port=27028)
    logger.setup_tracking()
    
    tags = setup_experiment_tags()
    with logger(tags=tags):
        logger.log_parameter("neurons", args.neurons)
        logger.log_parameter("train epochs", args.train_epochs)
        # logger.log_parameter("inter train", inter_train)
        # logger.log_parameter("random growth", random_growth)
        # logger.log_parameter("restrict actions", restrict_actions)
        # logger.log_parameter("estimate dependencies", estimate_dependencies)

        grow_network(
            model=model,
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            val_dataloader=val_dataloader,
            inter_train_dataloader=inter_train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=torch.nn.CrossEntropyLoss(),
            steps=args.iters,
            train_epochs=args.train_epochs,
            # inter_train=args.inter_train,
            # random_growth=args.random_growth,
            # restrict_actions=args.restrict_actions,
            # estimate_dependencies=args.estimate_dependencies,
            logger=logger,
            verbose=False,
        )

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # p1 = ax1.plot(net.hist_loss_dev, label="development loss")
    # p2 = ax2.plot(net.hist_acc_dev, label="development accuracy")
    # # p3 = ax2.plot(net.hist_acc_val, label="validation accuracy")
    # # p4 = ax1.scatter(indices, loss_train, marker='o', label="train loss")
    # # p5 = ax1.scatter(indices, loss_test, marker='o', label="test loss")
    # # p6 = ax2.scatter(indices, acc_train, marker='^', label="train accuracy")
    # p7 = ax2.scatter(indices, acc_test, marker="^", label="test accuracy")
    # plots = p1 + p2 + [p7]
    # labels = [p.get_label() for p in plots]
    # ax1.legend(plots, labels)
    # ax1.set_xlabel("epochs of intermediate training")
    # ax1.set_ylabel("loss")
    # ax2.set_ylabel("accuracy")
    # plt.show()
    # print()

    # TODO: check correct data partitions
    # TODO: remove print statements
    # TODO: fix documentation
    # TODO: profile memory
    # TODO: compare with simple model
    # TODO: test with cifar
