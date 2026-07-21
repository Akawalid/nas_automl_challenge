import torch
from functools import partial
from time import time
import numpy as np
import operator
import networkx as nx
import os
import json
import gc

from gromo.utils.utils import global_device, set_device
from gromo.utils.locate_dependence import calculate_dependency
from gromo.containers.growing_graph_network import GrowingGraphNetwork

from helpers import (
    evaluate_model,
    topk_accuracy,
    show_time,
)

from trainer import Trainer

class NAS:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor

        You can modify or add anything into the metadata that you wish,
        if you want to pass messages between your classes,
    """
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader 
        self.valid_loader = valid_loader 
        self.metadata = metadata
        self.clock = clock
        self.set_metadata()
    
    def find_number_of_datasets(self):
        datasets = [name for name in os.listdir("datasets")]
        for i, data_path in enumerate(datasets):
            with open(os.path.join("datasets", data_path, 'metadata'), "r") as f:
                meta = json.load(f)
            if meta["codename"] == self.metadata["codename"]:
                index = i
                break
        return len(datasets), index
    
    def set_metadata(self):
        nb_of_datasets, index = self.find_number_of_datasets()
        self.metadata.update({
            # model arguments
            "no_cuda": False,
            "initial_hidden_size": 50,

            # training arguments
            # "batch_size": 64,
            "num_steps": 20,
            "training_epochs": 10,
            "growing_maximum_added_neurons": 20,
            "estimate_dependencies": True,
            "training_threshold": 1e-5,

            # datasets
            "nb_of_datasets_total": nb_of_datasets,
            "current_dataset_index" : index,
            "nb_of_datasets_left": nb_of_datasets - index,
        })

    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """

    def search(self):
        start_time = time()

        # set the device 
        if self.metadata.get('no_cuda'):
            set_device(torch.device("cpu"))
        self.device: torch.device = global_device()


        # get the input/output shape from metadata
        input_shape = self.metadata['input_shape'][1:]
        output_shape = self.metadata['num_classes']
        

        # Initialize loss function
        self.loss_function_train = torch.nn.CrossEntropyLoss()
        self.top_1_accuracy = partial(topk_accuracy, k=1)


        # Initialize the model 
        model = GrowingGraphNetwork(
            in_features=np.prod(input_shape),
            out_features=output_shape,
            neurons=self.metadata["growing_maximum_added_neurons"],
            layer_type="linear",
            loss_fn=self.loss_function_train,
            device=self.device,
        )
        node_attributes = {
            "type": model.layer_type,
            "size": self.metadata["initial_hidden_size"],
            "activation": "selu",
        }
        model.dag.add_node_with_two_edges(model.dag.root, "1", model.dag.end, node_attributes=node_attributes)
        print("Starting model:")
        print(model.dag)

        
        # NAS loop
        estimated_steps_remaining = 1
        step = 0
        while estimated_steps_remaining >=1:
            step += 1
            step_start_time = time()
          
            # ----- TRAINING EPOCH -----

            trainer = Trainer(model, device=self.device,
                    train_dataloader=self.train_loader,
                    valid_dataloader=self.valid_loader,
                    metadata=self.metadata,
                    clock=self.clock)
            
            model = trainer.train()

              
            # ----- GROWTH EPOCH -----
            self.growth_step(model=model)
            val_loss, val_accuracy = evaluate_model(
                    model=model,
                    loss_function=self.loss_function_train,
                    aux_loss_function=self.top_1_accuracy,
                    dataloader=self.valid_loader,
                    device=self.device,
                )
            
            train_loss, train_accuracy = evaluate_model(
                    model=model,
                    loss_function=self.loss_function_train,
                    aux_loss_function=self.top_1_accuracy,
                    dataloader=self.train_loader,
                    device=self.device,
                )
            
            approx_time_per_step = (time() - step_start_time) / step
            time_left = self.clock.check() / self.metadata["nb_of_datasets_left"] - 120
            estimated_steps_remaining = (time_left / approx_time_per_step) * 0.8

            # print(f"clock={self.clock.check()/3600} {time_left/3600=} datasets_left={self.metadata['nb_of_datasets_left']}")

            print("After growth Step {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | Train Loss: {:>2.6f} | Valid Loss: {:>2.6f} | T/Epoch: {:<7} | Steps left: {:<1}".format(
                step, self.metadata['num_steps'],
                train_accuracy * 100, val_accuracy * 100,
                train_loss, val_loss,
                show_time(approx_time_per_step),
                int(estimated_steps_remaining),
            ))
            

            # Early stopping
            if (
                self.metadata.get('training_threshold') is not None
                and train_loss < self.metadata['training_threshold']
            ):
                print(f"Training threshold reached at step {step}")
                break
    
        print(f"Total duration: {time() - start_time}")
        print(model)

        return model
    
    def growth_step(self, model):
        with torch.enable_grad():
            model.global_step += 1
           
            # Find new ways to grow the DAG
            generations = model.dag.define_next_actions()
            print(f"Starting with {len(generations)} actions")

            # Retrieve expressivity bottleneck and inputs on important nodes
            bottleneck, input_B = model.dag.calculate_bottleneck(generations, self.train_loader)

            bott_norms = {key: torch.linalg.norm(val) for key, val in bottleneck.items()}
            important_node = max(bott_norms.items(), key=operator.itemgetter(1))[0]
            print(
                f"Restricting action space to output node '{important_node}' with norm {bott_norms[important_node]}"
            )
            generations = model.restrict_action_space(generations, chosen_outputs=[important_node])

            if self.metadata["estimate_dependencies"] and len(generations) > 3:
                input_B = {name:value for name, value in input_B.items() if name not in nx.descendants(model.dag, important_node) and name != important_node}
                hsic = calculate_dependency(input_B, bottleneck[important_node], n_samples=1000)
                hsic_values = torch.tensor(list(hsic.values()), device=global_device(), dtype=float)
                percentile = torch.quantile(hsic_values, 0.9)
                dominant_inputs = [name for name, value in hsic.items() if value >= percentile]
                if len(dominant_inputs) > 0:
                    generations = model.restrict_action_space(generations, chosen_inputs=dominant_inputs)
                    print(f"Restricting action space to input nodes {dominant_inputs} with hsic above {percentile}")
            
            print(f"Evaluating {len(generations)} actions")
            
            # Get random batch
            X_train, Y_train = next(iter(self.train_loader)) # random batch from train set
            X_dev, Y_dev = next(iter(self.train_loader)) # random batch from train set
            X_val, Y_val = next(iter(self.valid_loader)) # random batch from validation set
            X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
            X_dev, Y_dev = X_dev.to(self.device), Y_dev.to(self.device)
            X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)

            # Execute all graph growth options
            model.execute_expansions(
                actions=generations,
                bottleneck=bottleneck,
                input_B=input_B,
                X_train=X_train,
                Y_train=Y_train,
                X_dev=X_dev,
                Y_dev=Y_dev,
                X_val=X_val,
                Y_val=Y_val,
                amplitude_factor=True,
                evaluate=True,
                verbose=False,
            )

            # Find option that generates minimum loss
            model.choose_growth_best_action(
                generations, use_bic=False, verbose=True
            )

            # Memory optimization
            gc.collect()
            torch.cuda.empty_cache()

