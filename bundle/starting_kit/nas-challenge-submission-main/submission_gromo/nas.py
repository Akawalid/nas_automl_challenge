import torch
import os
from functools import partial
from time import time
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split

from gromo.utils.utils import global_device, set_device

from helpers import (
    get_model_from_config,
    compute_statistics,
    evaluate_model,
    line_search,
    topk_accuracy,
    train,
    get_scheduler,
    create_parser,
    update_config_from_args,
    known_datasets,
    known_optimizers,
    known_schedulers,
    selection_methods,
    show_time,
    get_transforms,
)

from data_processor import DataProcessor, TorchDataset
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
    
    def set_metadata(self):
        self.metadata.update({
            # model arguments
            "model_config": None,
            "no_cuda": False,  

            # training arguments
            "batch_size": 64,
            "num_steps": 20,
            "training_epochs": 10,
            "optimizer": "adam",
            "lr": 0.001,
            "weight_decay": 0.0,
            "training_threshold": None,  

            #sechduler arguments
            "scheduler": "none",
            "warmup_epochs": 0,

            # growing arguments
            "epochs_per_growth": 10,
            "selection_method": "fo",
            "growing_batch_limit": -1,
            "growing_part": "all",
            "growing_numerical_threshold": 1e-5,
            "growing_statistical_threshold": 1e-3,
            "growing_maximum_added_neurons": 10,
            "growing_computation_dtype": "float32",
            "normalize_weights": False,
            "init_new_neurons_with_random_in_and_zero_out": False,

            # line search arguments
            "line_search_alpha": 0.1,
            "line_search_beta": 0.5,
            "line_search_max_iter": 20,
            "line_search_epsilon": 1e-7,
            "line_search_batch_limit": -1,
        })

    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """

    def search(self):
        
        #set up for config parser
        parser = create_parser()
        args = parser.parse_args()

        # with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as file:
        #     config = yaml.safe_load(file)
        config = {
            "model": "mlp",
            "activation": "relu",
            "number_hidden_layers": 3,
            "hidden_size": 128,
        }
        update_config_from_args(config,args)
        self.metadata['model_config'] = config

        start_time = time()

        # known_optimizers = {
        #     "sgd": torch.optim.SGD,
        #     "adam": torch.optim.Adam,
        # }

        # set the device 
        if self.metadata.get('no_cuda'):
            set_device(torch.device("cpu"))
        self.device: torch.device = global_device()


        # get the input/output shape from metadata
        input_shape = self.metadata['input_shape'][1:]
        output_shape = self.metadata['num_classes']

        #initialize the model 
        model = get_model_from_config(
            in_features=input_shape,
            out_features=output_shape,
            config = self.metadata['model_config'],  
            )
        print("Starting model:")
        print(model)
        
        #initialize loss function
        self.loss_function_train = torch.nn.CrossEntropyLoss(reduction="mean")
        self.loss_function_growth = torch.nn.CrossEntropyLoss(reduction="sum")
        self.top_1_accuracy = partial(topk_accuracy, k=1)

        # #initialize optimizer
        # if self.metadata['optimizer'] == "sgd":  #add into metadata
        #     optim_kwargs = {
        #         "lr": self.metadata['lr'],
        #         "momentum": 0.9,
        #         "weight_decay": self.metadata['weight_decay'],
        #     }
        # elif self.metadata['optimizer'] == "adamw":
        #     optim_kwargs = {
        #         "lr": self.metadata['lr'],
        #         "betas": (0.9, 0.99),
        #         "weight_decay": self.metadata['weight_decay'],
        #     }
        # elif self.metadata['optimizer'] == "adam":
        #     optim_kwargs = {
        #         "lr": self.metadata['lr'],
        #         "betas": (0.9, 0.99),
        #         "weight_decay": self.metadata['weight_decay'],
        #     }

        # optimizer = known_optimizers[self.metadata['optimizer']](model.parameters(), **optim_kwargs)
        
        # #initiliazing scheduler
        # scheduler = get_scheduler(
        #     scheduler_name=self.metadata['scheduler'],
        #     optimizer=optimizer,
        #     num_epochs=self.metadata['num_steps'],
        #     num_batches_per_epoch=len(self.train_loader),
        #     base_lr=self.metadata['lr'],
        #     warmup_epochs=self.metadata['warmup_epochs'],
        # )

        # growing dtype
        self.growing_dtype = torch.float32
        if self.metadata['growing_computation_dtype'] == "float64":
            self.growing_dtype = torch.float64
        elif self.metadata['growing_computation_dtype'] != "float32":
            raise ValueError(f"Unknown growing dtype: {self.metadata['growing_computation_dtype']}")

        # evaluate the model on the train, val and test sets
        train_loss, train_accuracy = evaluate_model(
            model=model,
            loss_function=self.loss_function_train,
            aux_loss_function=self.top_1_accuracy,
            dataloader=self.train_loader,
            device=self.device,
        )
        val_loss, val_accuracy = evaluate_model(
            model=model,
            loss_function=self.loss_function_train,
            aux_loss_function=self.top_1_accuracy,
            dataloader=self.valid_loader,
            device=self.device,
        )

        # def is_growth_epoch(step: int) -> bool:
        #     assert step > 0, "Step should be greater than 0"
        #     if self.metadata['epochs_per_growth'] == -1:
        #         return False
        #     else:
        #         return step % (self.metadata['epochs_per_growth'] + 1) == 0

        training_epochs = self.metadata['training_epochs']
        
        #initalizing loop
        last_updated_layer = -1
        for step in range(1, self.metadata['num_steps'] + 1):
            step_start_time = time()
            # if is_growth_epoch(step):
            # ----- GROWTH EPOCH -----
            self.growth_step(model=model, last_updated_layer=last_updated_layer)
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
            # else:
            # ----- TRAINING EPOCH -----

            trainer = Trainer(model, device=self.device,
                    train_dataloader=self.train_loader,
                    valid_dataloader=self.valid_loader,
                    metadata=self.metadata,
                    clock=self.clock)

            model = trainer.train()
            # val_loss, val_accuracy = evaluate_model(
            #         model=model,
            #         loss_function=self.loss_function_train,
            #         aux_loss_function=self.top_1_accuracy,
            #         dataloader=self.valid_loader,
            #         device=self.device,
            #     )
            
            train_loss, train_accuracy = evaluate_model(
                    model=model,
                    loss_function=self.loss_function_train,
                    aux_loss_function=self.top_1_accuracy,
                    dataloader=self.train_loader,
                    device=self.device,
                )
                
            # print("Before growth Step {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} |".format(
            #     step, self.metadata['num_steps'] + 1,
            #     train_accuracy * 100, val_accuracy * 100,
            #     show_time((time() - step_start_time) / (step))
            # ))
            

            # Early stopping
            if (
                self.metadata['training_threshold'] is not None
                and train_loss < self.metadata['training_threshold']
            ):
                print(f"Training threshold reached at step {step}")
                break
    
        print(f"Total duration: {time() - start_time}")
        print(model)

        return model
    
    def growth_step(self, model, last_updated_layer):
        with torch.enable_grad():

            # 1) Gather Statistics
            initial_val_loss, initial_val_accuracy = compute_statistics(
                growing_model=model,
                dataloader=self.valid_loader,
                loss_function=self.loss_function_growth,
                aux_loss_function=self.top_1_accuracy,
                batch_limit=self.metadata['growing_batch_limit'],
                device=self.device,
            )

            # 2) Compute optimal updates
            model.compute_optimal_updates(
                zero_delta=self.metadata['growing_part'] == "neuron",
                numerical_threshold=self.metadata['growing_numerical_threshold'],
                statistical_threshold=self.metadata['growing_statistical_threshold'],
                maximum_added_neurons=self.metadata['growing_maximum_added_neurons'],
                dtype=self.growing_dtype,
            )

            # 3) Select the layer to update
            if self.metadata['selection_method'] == "none":
                last_updated_layer = (last_updated_layer + 1) % len(
                    model._growing_layers
                )
                model.select_update(layer_index=last_updated_layer)
            elif self.metadata['selection_method'] == "fo":
                last_updated_layer = model.select_best_update()
            else:
                raise NotImplementedError("Growing the model is not implemented yet")
            print(f"Currently updated layer: {model.currently_updated_layer.name}")

            # (Optional) 4) Initialize the new neurons with random fan-in weights and zero fan-out weight
            #To be included ? 

            # 4) Compute the optimal gamma
            if not self.metadata['init_new_neurons_with_random_in_and_zero_out']:
                (
                    gamma,
                    val_loss,
                    val_accuracy,
                    gamma_history,
                    loss_history,
                    aux_loss_history,
                ) = line_search(
                    model=model,
                    dataloader=self.valid_loader,
                    loss_function=self.loss_function_growth,
                    aux_loss_function=self.top_1_accuracy,
                    batch_limit=self.metadata['growing_batch_limit'],
                    initial_loss=initial_val_loss,
                    first_order_improvement=model.currently_updated_layer.first_order_improvement,
                    alpha=self.metadata['line_search_alpha'],
                    beta=self.metadata['line_search_beta'],
                    max_iter=self.metadata['line_search_max_iter'],
                    epsilon=self.metadata['line_search_epsilon'],
                    device=self.device,
                )

                # 5) Apply the change
                model.currently_updated_layer.scaling_factor = gamma**0.5
                model.apply_change()
                model.reset_computation()
                if self.metadata['normalize_weights']:
                    model.normalise()

                # #Reset the optimizer after growing
                # optimizer = known_optimizers[self.metadata['optimizer']](
                #     model.parameters(), **optim_kwargs
                # )
                # scheduler.optimizer = optimizer



def main():

    # # load model config from YAML
    # with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as file:
    #     model_config = yaml.safe_load(file)

    # # load data from DataProcessor.py 
    # train_x = np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8)
    # train_y = np.random.randint(0, 10, size=(100,))
    # valid_x = np.random.randint(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
    # valid_y = np.random.randint(0, 10, size=(20,))
    # test_x  = np.random.randint(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)
    def load_cifar10_numpy():
        # Download training set
        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        train_x = np.array(cifar10_train.data, dtype=np.uint8)  # shape: (50000, 32, 32, 3)
        train_y = np.array(cifar10_train.targets, dtype=np.int64)  # shape: (50000,)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)
        
        # Download validation/test set
        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        # valid_x = np.array(cifar10_test.data[:5000], dtype=np.uint8)  # use half of test set for validation
        # valid_y = np.array(cifar10_test.targets[:5000], dtype=np.int64)

        test_x = np.array(cifar10_test.data, dtype=np.uint8)
        test_y = np.array(cifar10_test.targets, dtype=np.uint8)
        return train_x, train_y, valid_x, valid_y, test_x, test_y


    train_x, train_y, valid_x, valid_y, test_x, test_y = load_cifar10_numpy()
    
    # initial metadata for dataprocessor
    metadata = {
        'num_classes': 10,
        'codename': 'cifar10',  
        'input_shape': train_x.shape,  # HWC
        'time_remaining': 3600
    }

    processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, clock=None)
    train_loader, valid_loader, test_loader = processor.process()
    # input_shape = train_loader.dataset.data[0].shape

    # metadata with all necessary parameters / arguments from parser 
    metadata.update({
        # "codename": 'cifar10', 
        # "num_classes": 10,
        # "input_shape": input_shape,
        # "time_remaining": 3600,  

        # model arguments
        # "model_config": model_config,
        "config_path": "config.yaml",
        "no_cuda": False,  

        # training arguments
        "batch_size": 64,
        "num_steps": 20,
        "training_epochs": 10,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "training_threshold": None,  

        #sechduler arguments
        "scheduler": "none",
        "warmup_epochs": 0,

        # growing arguments
        "epochs_per_growth": 10,
        "selection_method": "fo",
        "growing_batch_limit": -1,
        "growing_part": "all",
        "growing_numerical_threshold": 1e-5,
        "growing_statistical_threshold": 1e-3,
        "growing_maximum_added_neurons": 10,
        "growing_computation_dtype": "float32",
        "normalize_weights": False,
        "init_new_neurons_with_random_in_and_zero_out": False,

        # line search arguments
        "line_search_alpha": 0.1,
        "line_search_beta": 0.5,
        "line_search_max_iter": 20,
        "line_search_epsilon": 1e-7,
        "line_search_batch_limit": -1,

    })

    # run NAS search
    nas = NAS(train_loader, valid_loader, metadata, clock=None)
    model = nas.search()

    print(" Final model returned by NAS.search():")
    print(model)


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

