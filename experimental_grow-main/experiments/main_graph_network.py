import argparse

import sys
import git
import matplotlib.pyplot as plt
import torch
from gromo.containers.growing_dag import GrowingDAG
from torch.utils.data import Dataset
from tqdm import tqdm

if '/home/tau/sdouka/codebase/experimental_grow' not in sys.path:
    sys.path.append('/home/tau/sdouka/codebase/experimental_grow')
if '/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow' not in sys.path:
    sys.path.append('/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow')

from wrappers.exp_graph_growing_net import ExpGrowingGraphNetwork
from tools.gpu_tracking import GpuTracker
from tools.datasets import get_dataset


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
    repo = git.Repo(search_parent_directories=True)
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

acc_test = []
indices = []


def grow_network(
    net: ExpGrowingGraphNetwork,
    trainset: Dataset,
    testset: Dataset,
    data_rng: torch.Generator,
    steps: int,
    inter_train: bool,
    train_epochs: int,
    # new_opt: bool,
    random_growth: bool,
    restrict_actions: bool,
    estimate_dependencies: bool,
    verbose: bool = False,
):
    tags = setup_experiment_tags()
    
    with net.logger(tags=tags):
        net.logger.log_parameter("neurons", args.neurons)
        net.logger.log_parameter("inter train", inter_train)
        net.logger.log_parameter("train epochs", train_epochs)
        net.logger.log_parameter("random growth", random_growth)
        net.logger.log_parameter("restrict actions", restrict_actions)
        net.logger.log_parameter("estimate dependencies", estimate_dependencies)

        for _ in tqdm(range(steps)):
            print("\nStep", net.global_step + 1)
            with GpuTracker(gpu_index=[tags["gpu_index"]], logger=net.logger) as tracker:
                net.grow_step(
                    train_dataset=trainset,
                    test_dataset=testset,
                    generator=data_rng,
                    inter_train=inter_train,
                    train_epochs=train_epochs,
                    random_growth=random_growth,
                    restrict_actions=restrict_actions,
                    estimate_dependencies=estimate_dependencies,
                    verbose=verbose,
                )
            # net.logger.log_all_metrics(step=net.global_step)
            # Temporary stats
            acc_test.append(net.acc_test)
            indices.append(len(net.hist_loss_dev))

            if verbose:
                print("\n********* NEW GRAPH *********")
                net.dag.draw()

            # try:
            #     graph = DAG_to_pyvis(net.network.G)
            #     pyvis_path = "tmp/graph_.html"
            #     with portalocker.Lock(pyvis_path, timeout=1) as fh:
            #         graph.save_graph(pyvis_path)
            #         net.logger.log_artifact(pyvis_path)
            # except Exception as error:
            #     print(error)
        # print(mlflow.MlflowClient().get_run(run.info.run_id).data)
    # net.logger.end_run()
    return net


if __name__ == "__main__":
    args = parse_args()

    trainset, _, testset = get_dataset("addnist", "data", split_train_val=0)
    in_features = 28 * 28 * 3
    out_features = 20

    net = ExpGrowingGraphNetwork(
        in_features=in_features,
        out_features=out_features,
        neurons=args.neurons,
        exp_name=args.exp_name,
        use_batch_norm=False, # TODO: fix bug with BatchNorm running average size
        loss_fn=torch.nn.CrossEntropyLoss(),
    )
    edges = [(net.dag.root, "1"), ("1", "2"), ("2", "3"), ("3", net.dag.end)]
    node_attributes = {
        net.dag.root: {
            "type": "linear",  # shows what follows
            "size": in_features,
            "activation": "flatten",
        },
        net.dag.end: {
            "type": "linear",
            "size": out_features,
        },
        "1": {"type": "linear", "size": 300, "activation": "selu"},
        "2": {"type": "linear", "size": 150, "activation": "selu"},
        "3": {"type": "linear", "size": 50, "activation": "selu"},
    }
    edge_attributes = {
        "type": "linear",
        "use_bias": True,
    }

    DAG_parameters = {}
    DAG_parameters["edges"] = edges
    DAG_parameters["node_attributes"] = node_attributes
    DAG_parameters["edge_attributes"] = edge_attributes
    dag = GrowingDAG(in_features=in_features, out_features=out_features, neurons=args.neurons, use_bias=True, use_batch_norm=False, DAG_parameters=DAG_parameters)
    net.dag = dag

    print(net.dag)
    # net.dag.draw()
    print(net.device)
    print()

    data_rng = torch.Generator()
    grow_network(
        net,
        trainset,
        testset,
        data_rng=data_rng,
        steps=args.iters,
        inter_train=args.inter_train,
        train_epochs=args.train_epochs,
        # new_opt=args.new_opt,
        random_growth=args.random_growth,
        restrict_actions=args.restrict_actions,
        estimate_dependencies=args.estimate_dependencies,
        verbose=False,
    )

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    p1 = ax1.plot(net.hist_loss_dev, label="development loss")
    p2 = ax2.plot(net.hist_acc_dev, label="development accuracy")
    # p3 = ax2.plot(net.hist_acc_val, label="validation accuracy")
    # p4 = ax1.scatter(indices, loss_train, marker='o', label="train loss")
    # p5 = ax1.scatter(indices, loss_test, marker='o', label="test loss")
    # p6 = ax2.scatter(indices, acc_train, marker='^', label="train accuracy")
    p7 = ax2.scatter(indices, acc_test, marker="^", label="test accuracy")
    plots = p1 + p2 + [p7]
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels)
    ax1.set_xlabel("epochs of intermediate training")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy")
    plt.show()
    print()

    # TODO: check correct data partitions
    # TODO: remove print statements
    # TODO: fix documentation
    # TODO: profile memory
    # TODO: compare with simple model
    # TODO: test with cifar
