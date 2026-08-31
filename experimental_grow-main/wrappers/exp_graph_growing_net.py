import copy
import json
import operator
import random
import os
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network as Pyvisnet
from torch.utils.data import DataLoader, Dataset, random_split

# from memory_profiler import profile as memprofile
# from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis
# from memory_profiler import LogFile


from gromo.config.loader import load_config
from gromo.containers.growing_dag import GrowingDAG, Expansion
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.modules.linear_growing_module import LinearMergeGrowingModule
from gromo.utils.locate_dependence import calculate_dependency

# from gromo.utils.dependence_estimator import calculate_dependency

from tools.logger import Logger
from tools.profiling import CustomProfile, profile_function
from tools.utils import (
    DAG_to_pyvis,
    mini_batch_gradient_descent,
    set_from_conf,
    evaluate_dataset,
    global_device,
)


class ExpGrowingGraphNetwork(GrowingGraphNetwork):
    """Growable DAG Network

    Parameters
    ----------
    in_features : int
        size of input features
    out_features : int
        size of output dimension
    loss_fn : torch.nn.Module
        loss function
    use_bias : bool, optional
        automatically use bias in the layers, by default True
    use_batch_norm : bool, optional
        use batch normalization on the last layer, by default False
    neurons : int, optional
        default number of neurons to add at each step, by default 20
    device : str | None, optional
        default device, by default None
    test_batch_size : int, optional
        batch size to use on the test set, by default 256
    exp_name : str, optional
        experiment name for logger, by default "Debug"
    with_profiler : bool, optional
        execute with profiling, by default False
    with_logger : bool, optional
        log results during execution, by default True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        loss_fn: torch.nn.Module,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        layer_type: str = "linear",
        input_shape: int = None,
        neurons: int = 20,
        device: str | None = None,
        test_batch_size: int = 256,
        exp_name: str = "Debug",
        with_profiler: bool = False,
        with_logger: bool = None,
    ) -> None:
        super(ExpGrowingGraphNetwork, self).__init__(
            in_features=in_features,
            out_features=out_features,
            loss_fn=loss_fn,
            neurons=neurons,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
            layer_type=layer_type,
            input_shape=input_shape,
            device=device,
        )
        self.test_batch_size = test_batch_size
        self.with_profiler = with_profiler

        if with_logger is None:
            with_logger = set_from_conf(self, "logging", True, setter=False)
        assert with_logger is not None
        self.logger = Logger(exp_name, enabled=with_logger)
        self.logger.setup_tracking()

        self.hist_loss_dev = []
        self.hist_acc_dev = []

    def reset_network(self) -> None:
        """Reset graph to empty"""
        super().reset_network()
        if "logger" in self.__dict__:
            self.logger.clear()

    def log_growth_info(self, x: torch.Tensor) -> None:
        """Log important metrics at the end of each step

        Parameters
        ----------
        x : torch.Tensor
            input features batch to infer model signature
        """
        self.logger.log_all_metrics(step=self.global_step)

        self.logger.log_metric("growth train loss", self.growth_loss_train, self.global_epoch)
        self.logger.log_metric("growth dev loss", self.growth_loss_dev, self.global_epoch)
        self.logger.log_metric("dev loss", self.loss_dev, self.global_epoch)
        self.logger.log_metric("growth val loss", self.growth_loss_val, self.global_epoch)
        self.logger.log_metric("val loss", self.loss_val, self.global_epoch)
        self.logger.log_metric("growth test loss", self.growth_loss_test, self.global_epoch)
        self.logger.log_metric("test loss", self.loss_test, self.global_epoch)
        
        self.logger.log_metric("growth train accuracy", self.growth_acc_train, self.global_epoch)
        self.logger.log_metric("growth dev accuracy", self.growth_acc_dev, self.global_epoch)
        self.logger.log_metric("dev accuracy", self.acc_dev, self.global_epoch)
        self.logger.log_metric("growth val accuracy", self.growth_acc_val, self.global_epoch)
        self.logger.log_metric("val accuracy", self.acc_val, self.global_epoch)
        self.logger.log_metric("growth test accuracy", self.growth_acc_test, self.global_epoch)
        self.logger.log_metric("test accuracy", self.acc_test, self.global_epoch)

        # model_copy = copy.deepcopy(self)
        # model_copy.to(self.device)
        # for param in model_copy.parameters():
        #     param.requires_grad = False
        # flops = FlopCountAnalysis(model_copy, x)
        # self.logger.log_metric("complexity/flops", flops.total(), self.global_epoch)
        # activations = ActivationCountAnalysis(model_copy, x)
        # self.logger.log_metric(
        #     "complexity/activations", activations.total(), self.global_epoch
        # )
        # del model_copy

        self.logger.log_metric(
            "complexity/nb of parameters",
            self.dag.count_parameters_all(),
            self.global_epoch,
        )
        # nb of parameters per edge
        for edge in self.dag.edges:
            params = self.dag.count_parameters([edge])
            self.logger.log_metric(
                f"complexity/nb of parameters at/layer {edge[0]}_{edge[1]}",
                params,
                self.global_epoch,
            )
        # in-degree and out-degree per node
        for node in self.dag.nodes:
            self.logger.log_metric(
                f"complexity/in-degree/node {node}",
                self.dag.in_degree(node),
                self.global_epoch,
            )
            self.logger.log_metric(
                f"complexity/out-degree/node {node}",
                self.dag.out_degree(node),
                self.global_epoch,
            )
            self.logger.log_metric(
                f"complexity/size/node {node}",
                self.dag.nodes[node]["size"],
                self.global_epoch,
            )

        with torch.no_grad():
            # Save model
            try:
                self.logger.log_pytorch_model(
                    self.dag, f"GrowableDAG step {self.global_step}", x
                )
            except Exception as error:
                print(f"[DAGNN Model] {error}")

        # Save growth history file
        dirname = "temp"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        try:
            with open(f"{dirname}/gh.json", "w") as f:
                json.dump(self.growth_history, f)
                f.flush()
                self.logger.log_artifact(f"{dirname}/gh.json")
        except Exception as error:
            print(f"[Growth History] {error}")

        # Save interactive graph
        try:
            graph = DAG_to_pyvis(self.dag)
            pyvis_path = f"{dirname}/graph_.html"
            graph.save_graph(pyvis_path)
            self.logger.log_artifact(pyvis_path)
        except Exception as error:
            print(f"[Interactive DAG] {error}")

    @profile_function
    def setup_train_datasets(
        self, train_dataset: Dataset, generator: torch.Generator
    ) -> tuple:
        """Split train dataset in three parts for train, development and validation

        Parameters
        ----------
        train_dataset : Dataset
            whole train dataset
        generator : torch.Generator
            random generator for dataset shuffling

        Returns
        -------
        tuple
            features and labels of train, development and validation datasets
        """
        split_length = len(train_dataset) // 3 # type: ignore
        residual_len = len(train_dataset) - split_length * 2 # type: ignore
        train_dataset, dev_dataset, val_dataset = random_split(
            train_dataset, (residual_len, split_length, split_length), generator
        )

        train_loader = DataLoader(
            train_dataset, len(train_dataset), shuffle=True, generator=generator
        )
        dev_loader = DataLoader(
            dev_dataset, len(dev_dataset), shuffle=True, generator=generator
        )
        val_loader = DataLoader(
            val_dataset, len(val_dataset), shuffle=True, generator=generator
        )
        # print("Length of the train_loader:", len(train_loader))
        # print("Length of the dev_loader:", len(dev_loader))
        # print("Length of the val_loader:", len(val_loader))

        # # This way does not call the transforms!
        # X_train, Y_train = trainset.dataset.data[trainset.indices].to(self.device), trainset.dataset.targets[trainset.indices].to(self.device)
        # X_dev, Y_dev = devset.dataset.data[devset.indices].to(self.device), devset.dataset.targets[devset.indices].to(self.device)
        # X_val, Y_val = valset.dataset.data[valset.indices].to(self.device), valset.dataset.targets[valset.indices].to(self.device)

        X_train, Y_train = next(iter(train_loader))
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        X_dev, Y_dev = next(iter(dev_loader))
        X_dev, Y_dev = X_dev.to(self.device), Y_dev.to(self.device)
        X_val, Y_val = next(iter(val_loader))
        X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)

        return X_train, Y_train, X_dev, Y_dev, X_val, Y_val

    @profile_function
    # @memprofile
    def expand_node(
        self,
        expansion: Expansion,
        bottlenecks: dict,
        activities: dict,
        parallel: bool = True,
        verbose: bool = True,
    ) -> list:
        """Increase block dimension by expanding node with more neurons
        Increase output size of incoming layers and input size of outgoing layers
        Train new neurons to minimize the expressivity bottleneck

        Parameters
        ----------
        expansion : Expansion
            object with expansion information
        bottlenecks : dict
            dictionary with node names as keys and their calculated bottleneck tensors as values
        activities : dict
            dictionary with node names as keys and their pre-activity tensors as values
        parallel : bool, optional
            take into account parallel connections, by default True
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        list
            bottleneck loss history
        list
            bottleneck loss history
        """
        loss_history = super().expand_node(
            expansion=expansion,
            bottlenecks=bottlenecks,
            activities=activities,
            parallel=parallel,
            verbose=verbose
        )
        # Log neurons bottleneck loss
        for next_node in expansion.next_nodes:
            self.logger.save_metric(
                f"growth/bottleneck final loss/node {next_node}",
                loss_history[-1],
            )
        return loss_history

    @profile_function
    # @memprofile
    def update_edge_weights(
        self,
        expansion: Expansion,
        bottlenecks: dict,
        activities: dict,
        verbose: bool = True,
    ) -> list:
        """Update weights of a single layer edge
        Train layer to minimize the expressivity bottleneck

        Parameters
        ----------
        expansion : Expansion
            object with expansion information
        bottlenecks : dict
            dictionary with node names as keys and their calculated bottleneck tensors as values
        activities : dict
            dictionary with node names as keys and their pre-activity tensors as values
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        list
            bottleneck loss history
        list
            bottleneck loss history
        """
        loss_history = super().update_edge_weights(
            expansion=expansion,
            bottlenecks=bottlenecks,
            activities=activities,
            verbose=verbose
        )
        self.logger.save_metric(
            f"growth/bottleneck final loss/node {expansion.next_nodes}",
            loss_history[-1],
        )
        return loss_history

    @profile_function
    # @memprofile
    def find_amplitude_factor(
        self,
        net: GrowingDAG,
        x: torch.Tensor,
        y: torch.Tensor,
        node_module: LinearMergeGrowingModule,
        next_node_modules: list[LinearMergeGrowingModule] = None,
    ) -> float:
        """Find amplitude factor with line search

        Parameters
        ----------
        net : GrowingDAG
            network of interest
        x : torch.Tensor
            input features batch
        y : torch.Tensor
            true labels batch
        node_module : LinearMergeGrowingModule
            node module to be extended or node module at the end of the edge in case of single edge
        next_node_modules : list[LinearMergeGrowingModule], optional
            next node modules of module to be extended, leave empty in case of single edge, by default None

        Returns
        -------
        float
            amplitude factor that minimizes overall loss
        """
        factor = super().find_amplitude_factor(
            net=net,
            x=x,
            y=y,
            node_module=node_module,
            next_node_modules=next_node_modules,
        )
        self.logger.save_metric(
            f"growth/amplitude factor/node {node_module._name}",
            factor,
        )
        if next_node_modules is not None:
            for next_node_module in next_node_modules:
                self.logger.save_metric(
                    f"growth/amplitude factor/node {next_node_module._name}",
                    factor,
                )
        return factor

    @profile_function
    def inter_training(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        epochs: int = 100,
        lrate: float = 1e-3,
        return_eval: bool = False,
        verbose: bool = True,
    ) -> (
        tuple[list[float], list[float], list[float]]
        | tuple[
            list[float], list[float], list[float], list[float], list[float], list[float]
        ]
    ):
        """Perform training on the model
        Batch gradient descent with AdamW with lrate of 1e-3 and no weight decay
        Log train and validation metrics with logger

        Parameters
        ----------
        x : torch.Tensor
            train input features batch
        y : torch.Tensor
            train true labels batch
        x1 : torch.Tensor
            validation input features batch
        y1 : torch.Tensor
            validation true labels batch
        epochs : int, optional
            maximum number of epochs, by default 500
        lrate : float, optional
            learning rate, by default 1e-3
        return_eval : bool, optional
            return evaluation metrics on the validation set, by default False
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        tuple[list[float], list[float], list[float]] | tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
            train loss history, train accuracy history, train f1-score history
        """

        f1score_history = []
        loss_val_history, acc_val_history, f1score_val_history = [], [], []

        def eval_fn() -> None:
            val_acc, val_loss, val_f1score = self.evaluate(
                x1, y1, with_f1score=True
            )
            _, _, dev_f1score = self.evaluate(x, y, with_f1score=True)
            f1score_history.append(dev_f1score)
            loss_val_history.append(val_loss)
            acc_val_history.append(val_acc)
            f1score_val_history.append(val_f1score)
            self.logger.log_metric(
                "Intermediate training/val loss",
                val_loss,
                self.global_epoch,
            )
            self.logger.log_metric(
                "Intermediate training/val accuracy",
                val_acc,
                self.global_epoch,
            )
            self.global_epoch += 1

        loss_history, acc_history = mini_batch_gradient_descent(
            model=self,
            cost_fn=self.loss_fn,
            X=x,
            Y=y,
            lrate=lrate,
            batch_size=256,
            max_epochs=epochs,
            fast=False,
            eval_fn=eval_fn,
            verbose=verbose,
        )

        self.loss_dev = loss_history[-1]
        self.acc_dev = acc_history[-1]

        # Log intermediate training
        for i in range(len(loss_history)):
            self.logger.log_metric(
                "Intermediate training/dev loss",
                loss_history[i],
                self.global_epoch - epochs + i,
            )
            self.logger.log_metric(
                "Intermediate training/dev accuracy",
                acc_history[i],
                self.global_epoch - epochs + i,
            )

        if return_eval:
            return (
                loss_history,
                acc_history,
                f1score_history,
                loss_val_history,
                acc_val_history,
                f1score_val_history,
            )

        return loss_history, acc_history, f1score_history

    @profile_function
    def calculate_bottleneck(
        self, actions: list[Expansion], dataloader: torch.utils.data.DataLoader,
    ) -> tuple[dict, dict]:
        # Handle empty graph case
        constant_module = False
        if self.dag.is_empty():
            # Create constant module if the graph is empty
            constant_module = True
            edge_attributes = {"type": "L", "use_bias": self.use_bias, "constant": True}
            self.dag.add_direct_edge("start", "end", edge_attributes)

        # Find nodes of interest
        prev_node_modules = set()
        next_node_modules = set()
        for expansion in actions:
            prev_nodes = expansion.previous_nodes
            next_nodes = expansion.next_nodes

            if not isinstance(prev_nodes, list):
                prev_nodes = [prev_nodes]
            if not isinstance(next_nodes, list):
                next_nodes = [next_nodes]

            prev_node_modules.update(prev_nodes)
            next_node_modules.update(next_nodes)

        # Add hooks on node modules of interest
        prev_node_modules = self.dag.get_node_modules(prev_node_modules)
        next_node_modules = self.dag.get_node_modules(next_node_modules)
        # for node_module in prev_node_modules:
        #     node_module.store_activity = True
        # for node_module in next_node_modules:
        #     node_module.init_computation()
        self.dag.init_computation()

        # Forward - Backward step
        for X_train, Y_train in dataloader:
            self.zero_grad()
            pred = self(X_train)
            loss = self.loss_fn(pred, Y_train)
            loss.backward()
            self.dag.update_computation()

        input_B = {}
        bottleneck = {}

        self.dag.compute_optimal_delta()
        # Update tensors
        # for node_module in next_node_modules:
        #     assert node_module.previous_tensor_s is not None
        #     assert node_module.previous_tensor_m is not None
        #     node_module.previous_tensor_s.update()
        #     node_module.previous_tensor_m.update()

        #     # Compute optimal possible updates
        #     deltas = node_module.compute_optimal_delta(update=True, return_deltas=True)

        for node_module in next_node_modules:
            # Compute expressivity bottleneck
            bottleneck[node_module._name] = (
                node_module.projected_v_goal().clone().detach()
            )  # (batch_size, out_features)

            # # Log possible optimal updates
            # assert deltas is not None
            # assert node_module.pre_activity is not None
            # assert node_module.pre_activity.grad is not None
            # for i, edge_module in enumerate(node_module.previous_modules):
            #     self.logger.log_metric_with_stats(
            #         f"growth/possible update/edge {edge_module._name}/weight",
            #         deltas[i][0],
            #         self.global_step,
            #     )
            #     if edge_module.use_bias:
            #         self.logger.log_metric_with_stats(
            #             f"growth/possible update/edge {edge_module._name}/bias",
            #             deltas[i][1],
            #             self.global_step,
            #         )
            # del deltas
            self.logger.log_metric_with_stats(
                f"growth/desired update/node {node_module._name}",
                node_module.pre_activity.grad,
                self.global_step,
            )
            self.logger.log_metric_with_stats(
                f"growth/bottleneck/node {node_module._name}",
                bottleneck[node_module._name],
                self.global_step,
            )

            if constant_module:
                assert torch.all(
                    bottleneck[node_module._name] == node_module.pre_activity.grad
                ), "Graph is empty and the bottleneck should be the same as the pre_activity gradient. Expected: {node_module.pre_activity.grad} Found: {bottleneck[node_module._name]}"

            # # Reset tensors and remove hooks
            # node_module.reset_computation()
        self.dag.reset_computation()

        # Retrieve input activities
        for node_module in prev_node_modules:
            assert node_module.activity is not None
            # Save input activity of input layers
            input_B[node_module._name] = node_module.activity.clone().detach()

            # Reset tensors and remove hooks
            node_module.store_activity = False
            # node_module.delete_update()

        # bottleneck = (
        #     torch.cat((bottleneck.values()), dim=0).detach().clone()
        # )  # (batch_size, total_out_features)
        # concatenated_input_B = torch.cat(list(input_B.values()), dim=0)#.clone().detach()

        # Reset all hooks
        for node_module in self.dag.get_all_node_modules():
            if node_module in next_node_modules:
                for parallel_module in node_module.previous_modules:
                    parallel_module.reset_computation()
                    # DO NOT delete updates
                    # parallel_module.delete_update(include_previous=False)
            # Delete activities
            node_module.delete_update()

        if constant_module:
            # Remove constant module if needed
            self.dag.remove_direct_edge(self.dag.root, self.dag.end)

        return bottleneck, input_B

    @profile_function
    # @memprofile
    def grow_step(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        generator: torch.Generator,
        amplitude_factor: bool = True,
        inter_train: bool = True,
        train_epochs: int = 100,
        restrict_actions: bool = True,
        estimate_dependencies: bool = True,
        random_growth: bool = False,
        bic_criterion: bool = False,
        verbose: bool = True,
    ) -> None:
        """Increase the size of the DAG network as one step
        Perform all possible growth actions and choose greedily the one that minimizes the validation loss
        Log important metrics

        Parameters
        ----------
        train_dataset : Dataset
            training dataset object
        test_dataset : Dataset
            test dataset object
        generator : torch.Generator
            random generator for dataset shuffling
        amplitude_factor : bool, optional
            apply amplitude factor to the new neurons, by default True
        inter_train : bool, optional
            train the network after growth, by default True
        train_epochs : int, optional
            number of epochs to train the network between expansions, by default 100
        restrict_actions : bool, optional
            restrict action space to accelerate growth, by default True
        estimate_dependencies : bool, optional
            restrict possible inputs by estimating dependencies, by default True
        random_growth : bool, optional
            choose the expansion at random, by default False
        bic_criterion : bool, optional
            use BIC to select the network expansion, by default False
        verbose : bool, optional
            print info, by default True
        """

        self.global_step += 1

        # Find new ways to grow the DAG
        generations = self.dag.define_next_actions()
        self.logger.log_metric("complexity/nb of actions", len(generations), self.global_step)
        if verbose:
            print(f"{generations=}")

        # Split train dataset into 3 parts
        X_train, Y_train, X_dev, Y_dev, X_val, Y_val = self.setup_train_datasets(
            train_dataset=train_dataset, generator=generator
        )
        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=1024)
        test_loader = DataLoader(test_dataset, self.test_batch_size)

        # Retrieve expressivity bottleneck and inputs on important nodes
        bottleneck, input_B = self.dag.calculate_bottleneck(generations, train_dataloader)

        if random_growth:
            generations = [random.choice(generations)]
        elif restrict_actions:
            bott_norms = {key: torch.linalg.norm(val) for key, val in bottleneck.items()}
            important_node = max(bott_norms.items(), key=operator.itemgetter(1))[0]
            if verbose:
                print(
                    f"Restricting action space to node {important_node} with norm {bott_norms[important_node]}"
                )
            generations = self.restrict_action_space(generations, chosen_outputs=[important_node])
            self.logger.log_metric(f"actions/important node", important_node if important_node!=self.dag.end else -1, self.global_step)

            if estimate_dependencies and len(generations) > 3:
                input_B = {name:value for name, value in input_B.items() if name not in nx.descendants(self.dag, important_node) and name != important_node}
                hsic = calculate_dependency(input_B, bottleneck[important_node], n_samples=1000)
                hsic_values = torch.tensor(list(hsic.values()), device=global_device(), dtype=float)
                percentile = torch.quantile(hsic_values, 0.9)
                dominant_inputs = [name for name, value in hsic.items() if value >= percentile]
                generations = self.restrict_action_space(generations, chosen_inputs=dominant_inputs)
                for name, value in hsic.items():
                    self.logger.log_metric(f"actions/HSIC/node {name}", value, self.global_step)
                self.logger.log_metric("actions/HSIC 0.9 quantile", percentile, self.global_step)

        self.logger.log_metric("complexity/nb of tested actions", len(generations), self.global_step)
        # Execute all graph growth options
        self.execute_expansions(
            generations,
            bottleneck,
            input_B,
            X_train,
            Y_train,
            X_dev,
            Y_dev,
            X_val,
            Y_val,
            amplitude_factor=amplitude_factor,
            verbose=verbose,
        )

        # Find option that generates minimum loss
        self.choose_growth_best_action(
            generations, use_bic=bic_criterion, verbose=verbose
        )

        # Safety check
        self.dag.reset_computation()
        self.dag.delete_update()

        # Intermediate training
        if inter_train:
            self.growth_acc_test, self.growth_loss_test = evaluate_dataset(self, test_loader, self.loss_fn)
            hist_loss_dev, hist_acc_dev, _ = self.inter_training(
                torch.cat([X_train, X_dev], dim=0),
                torch.cat([Y_train, Y_dev], dim=0),
                X_val,
                Y_val,
                epochs=train_epochs,
                verbose=verbose,
            )

            ########## TEMPORARY SOLUTION ##########
            self.hist_loss_dev.extend(hist_loss_dev)
            self.hist_acc_dev.extend(hist_acc_dev)
            #######################################

        # Evaluation
        self.acc_val, self.loss_val = self.evaluate(X_val, Y_val)
        self.acc_test, self.loss_test = evaluate_dataset(self, test_loader, self.loss_fn)

        self.log_growth_info(X_train)

    def choose_growth_best_action(
        self, options: list[Expansion], use_bic: bool = False, verbose: bool = False
    ) -> None:
        """Choose the growth action with the minimum validation loss greedily
        Log average metrics of the current growth step
        Reconstruct chosen graph and discard the rest

        Parameters
        ----------
        options : list[Expansion]
            dictionary with all possible graphs and their statistics
        use_bic : bool, optional
            use BIC to select the network expansion, by default False
        verbose : bool, optional
            print info, by default False
        """

        # TODO: reinit metrics
        # DF["train loss reduction"] = self.loss_train - DF["loss_train"]
        # DF["dev loss reduction"] = self.loss_dev - DF["loss_dev"]
        # DF["val loss reduction"] = self.loss_val - DF["loss_val"]
        # self.log_metric_stats(
        #     "growth actions/average train loss reduction",
        #     DF["train loss reduction"].mean(),
        # )
        # self.log_metric_stats(
        #     "growth actions/average dev loss reduction", DF["dev loss reduction"].mean()
        # )
        # self.log_metric_stats(
        #     "growth actions/average val loss reduction", DF["val loss reduction"].mean()
        # )

        # Greedy choice based on validation loss
        selection = {}
        if use_bic:
            for index, expansion in enumerate(options):
                selection[index] = expansion.metrics["BIC"]
        else:
            for index, expansion in enumerate(options):
                selection[index] = expansion.metrics["loss_val"]
        
        best_ind = min(selection.items(), key=operator.itemgetter(1))[0]

        # Reconstruct graph
        best_option = options[best_ind]
        del options

        if verbose:
            print("Chose option", best_ind, best_option)

        # for metric, value in best_option.stats.items():
        #     self.log_metric_stats(metric, value)
        # best_option.stats.clear()
        del self.dag
        self.dag = copy.copy(best_option.dag)
        self.growth_history = best_option.growth_history
        self.growth_loss_train = best_option.metrics["loss_train"]
        self.growth_loss_dev = best_option.metrics["loss_dev"]
        self.growth_loss_val = best_option.metrics["loss_val"]
        self.growth_acc_train = best_option.metrics["acc_train"]
        self.growth_acc_dev = best_option.metrics["acc_dev"]
        self.growth_acc_val = best_option.metrics["acc_val"]
        del best_option
    
    def evaluate(self, x, y, with_f1score=False):
        return self.dag.evaluate(x=x, y=y, loss_fn=self.loss_fn, with_f1score=with_f1score)

    def draw(self) -> None:
        """
        Draw graph on plot
        """
        plt.figure()
        G = copy.deepcopy(self.dag)

        try:
            pos = nx.planar_layout(G)
        except:
            pos = nx.shell_layout(G)

        for edge in G.edges:
            if G.edges[edge]["type"] == "convolution":
                G.edges[edge]["style"] = "dashed"
            if G.edges[edge]["type"] == "linear":
                G.edges[edge]["style"] = "solid"

            nx.draw_networkx_edges(G, pos, [edge], style=G.edges[edge]["style"])

        for node in G.nodes:
            if G.nodes[node]["type"] == "linear":
                G.nodes[node]["color"] = "#42cafd"
                G.nodes[node]["shape"] = "o"
            elif G.nodes[node]["type"] == "convolution":
                G.nodes[node]["color"] = "#edd83d"
                G.nodes[node]["shape"] = "s"

            if node == self.dag.root:
                G.nodes[node]["color"] = "#09bc8a"
            elif node == self.dag.end:
                G.nodes[node]["color"] = "#f55536"

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_shape=G.nodes[node]["shape"],
                node_color=G.nodes[node]["color"],
            )

        # nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        plt.show()

    def animate(self) -> Pyvisnet:
        """Create animated version of DAG with pyvis Network

        Returns
        -------
        Pyvisnet
            pyvis network
        """

        return DAG_to_pyvis(self)
