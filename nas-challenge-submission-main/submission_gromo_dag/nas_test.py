import torch
from functools import partial
from time import time
import numpy as np
import torchvision
import operator
import networkx as nx
from sklearn.model_selection import train_test_split

from gromo.utils.utils import global_device, set_device
from gromo.utils.locate_dependence import calculate_dependency
from gromo.containers.growing_graph_network import GrowingGraphNetwork

from helpers import (
    evaluate_model,
    topk_accuracy,
    show_time,
    get_transforms,
)

from data_processor import DataProcessor, TorchDataset
from trainer import Trainer
from nas import NAS

class NASTEST:
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
    
    def set_metadata(self):
        self.metadata.update({
            # model arguments
            "no_cuda": False,
            "initial_hidden_size": 50,

            # training arguments
            # "batch_size": 64,
            "num_steps": 5,
            "training_epochs": 10,
            "growing_maximum_added_neurons": 10,
            "estimate_dependencies": True,
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


        # evaluate the model on the train, val and test sets
        # train_loss, train_accuracy = evaluate_model(
        #     model=model,
        #     loss_function=self.loss_function_train,
        #     aux_loss_function=self.top_1_accuracy,
        #     dataloader=self.train_loader,
        #     device=self.device,
        # )
        # val_loss, val_accuracy = evaluate_model(
        #     model=model,
        #     loss_function=self.loss_function_train,
        #     aux_loss_function=self.top_1_accuracy,
        #     dataloader=self.valid_loader,
        #     device=self.device,
        # )
        # print("Before growth Step {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% |".format(
        #     0, self.metadata['num_steps'] + 1,
        #     train_accuracy * 100, val_accuracy * 100,
        # ))

        
        # NAS loop
        for step in range(1, self.metadata['num_steps'] + 1):
            step_start_time = time()
          
            # ----- TRAINING EPOCH -----

            trainer = Trainer(model, device=self.device,
                    train_dataloader=self.train_loader,
                    valid_dataloader=self.valid_loader,
                    metadata=self.metadata,
                    clock=self.clock)
            
            model = trainer.train()

            
            # train_loss, train_accuracy = evaluate_model(
            #         model=model,
            #         loss_function=self.loss_function_train,
            #         aux_loss_function=self.top_1_accuracy,
            #         dataloader=self.train_loader,
            #         device=self.device,
            #     )
                
            # print("Before growth Step {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} |".format(
            #     step, self.metadata['num_steps'] + 1,
            #     train_accuracy * 100, val_accuracy * 100,
            #     show_time((time() - step_start_time) / (step))
            # ))

              
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
            
            print("After growth Step {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} |".format(
                step, self.metadata['num_steps'],
                train_accuracy * 100, val_accuracy * 100,
                show_time((time() - step_start_time) / (step))
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
            
            # # Get random batch
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


def main():
    def load_cifar10_numpy():
        # Download training set
        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        train_x = np.array(cifar10_train.data, dtype=np.uint8)  # shape: (50000, 32, 32, 3)
        train_y = np.array(cifar10_train.targets, dtype=np.int64)  # shape: (50000,)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)
        
        # Download validation/test set
        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

        test_x = np.array(cifar10_test.data, dtype=np.uint8)
        test_y = np.array(cifar10_test.targets, dtype=np.uint8)
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    train_x, train_y, valid_x, valid_y, test_x, test_y = load_cifar10_numpy()
    
    # initial metadata for dataprocessor
    metadata = {
        'num_classes': 20,
        'codename': 'Gutenberg',  
        'input_shape': train_x.shape,  # HWC
        'time_remaining': 200
    }

    processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, clock=None)
    train_loader, valid_loader, test_loader = processor.process()


    # run NAS search
    nas = NAS(train_loader, valid_loader, metadata, clock=None)
    model = nas.search()

    print(" Final model returned by NAS.search():")
    print(model.dag)


    base_tf = torchvision.transforms.v2.Compose(get_transforms(metadata['codename'])[0])
    test_dataset = TorchDataset(test_x, test_y, transform=base_tf)
    full_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

    test_loss, test_accuracy = evaluate_model(
        model=model,
        loss_function=nas.loss_function_train,
        aux_loss_function=nas.top_1_accuracy,
        dataloader=full_test_loader,
        device=nas.device,
    )
    print(f"Test loss {test_loss} Test accuracy {test_accuracy}")

if __name__ == "__main__":
    main()

