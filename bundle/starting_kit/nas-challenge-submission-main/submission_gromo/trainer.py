import torch
import os
import time
import numpy as np
import torch.nn as nn

from sklearn.metrics import accuracy_score


from helpers import show_time
from data_processor import DataProcessor
# from nas import NAS

class Trainer:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The Trainer class will receive the following inputs
        * model: The model returned by your NAS class
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor or NAS classes
    """
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

        # define training parameters
        self.epochs = metadata['training_epochs']
        # self.optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=3e-4)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        self.criterion = nn.CrossEntropyLoss()
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scheduler = None

    """
    ====================================================================================================================
    TRAIN ==============================================================================================================
    ====================================================================================================================
    The train function will define how your model is trained on the train_dataloader.
    Output: Your *fully trained* model
    
    See the example submission for how this should look
    """
    def train(self):
        if torch.cuda.is_available():
            self.model.cuda()
        t_start = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            labels, predictions = [], []
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model.forward(data)

                # store labels and predictions to compute accuracy
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            train_acc = accuracy_score(labels, predictions)
            valid_acc = self.evaluate()
            if (epoch + 1) % 5 == 0 or epoch == self.epochs or self.epochs <= 10:
                print("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} |".format(
                    epoch + 1, self.epochs,
                    train_acc * 100, valid_acc * 100,
                    show_time((time.time() - t_start) / (epoch + 1))
                ))
        print("  Total runtime: {}".format(show_time(time.time() - t_start)))
        return self.model

    # print out the model's accuracy over the valid dataset
    # (this isn't necessary for a submission, but I like it for my training logs)
    def evaluate(self):
        self.model.eval()
        labels, predictions = [], []
        for data, target in self.valid_dataloader:
            data = data.to(self.device)
            output = self.model.forward(data)
            labels += target.cpu().tolist()
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return accuracy_score(labels, predictions)


    """
    ====================================================================================================================
    PREDICT ============================================================================================================
    ====================================================================================================================
    The prediction function will define how the test dataloader will be passed through your model. It will receive:
        * test_dataloader created by your DataProcessor
    
    And expects as output:
        A list/array of predicted class labels of length=n_test_datapoints, i.e, something like [0, 0, 1, 5, ..., 9] 
    
    See the example submission for how this should look.
    """

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        for data in test_loader:
            data = data.to(self.device)
            output = self.model.forward(data)
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions

import torchvision

def main():

    # load model config from YAML
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as file:
        model_config = yaml.safe_load(file)

    
    # load data from DataProcessor.py 
    

    def load_cifar10_numpy():
        # Download training set
        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        train_x = np.array(cifar10_train.data, dtype=np.uint8)  # shape: (50000, 32, 32, 3)
        train_y = np.array(cifar10_train.targets, dtype=np.int64)  # shape: (50000,)

        # Download validation/test set
        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        valid_x = np.array(cifar10_test.data[:5000], dtype=np.uint8)  # use half of test set for validation
        valid_y = np.array(cifar10_test.targets[:5000], dtype=np.int64)

        test_x = np.array(cifar10_test.data[5000:], dtype=np.uint8)  # other half for test
        return train_x, train_y, valid_x, valid_y, test_x


    train_x, train_y, valid_x, valid_y, test_x = load_cifar10_numpy()


    #train_x = np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8)
    #train_y = np.random.randint(0, 10, size=(100,))
    #valid_x = np.random.randint(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
    #valid_y = np.random.randint(0, 10, size=(20,))
    #test_x  = np.random.randint(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)


    # initial metadata for dataprocessor
    init_metadata = {
        'num_classes': 10,
        'codename': 'cifar10',  
        'input_shape': (100, 32, 32, 3),  # HWC
        'time_remaining': 3600
    }

    processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, init_metadata, clock=None)
    train_loader, valid_loader, test_loader = processor.process()
    input_shape = train_loader.dataset.data[0].shape

    # check shape of one batch
    print("Train batch shape:", next(iter(train_loader))[0].shape)
    print("Valid batch shape:", next(iter(valid_loader))[0].shape)
    print("Test batch shape:", next(iter(test_loader))[0].shape)

    # metadata with all necessary parameters / arguments from parser 
    metadata = {
        "codename": 'cifar10', 
        "num_classes": 10,
        "input_shape": input_shape,
        "time_remaining": 3600,  

        # model arguments
        "model_config": model_config,
        "config_path": "config.yaml",
        "no_cuda": False,  

        # training arguments
        "batch_size": 64,
        "num_steps": 10,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "training_threshold": None,  

        #sechduler arguments
        "scheduler": "none",
        "warmup_epochs": 0,

        # growing arguments
        "epochs_per_growth": -1,
        "selection_method": "none",
        "growing_batch_limit": -1,
        "growing_part": "all",
        "growing_numerical_threshold": 1e-5,
        "growing_statistical_threshold": 1e-3,
        "growing_maximum_added_neurons": 5,
        "growing_computation_dtype": "float32",
        "normalize_weights": False,
        "init_new_neurons_with_random_in_and_zero_out": False,

        # line search arguments
        "line_search_alpha": 0.1,
        "line_search_beta": 0.5,
        "line_search_max_iter": 20,
        "line_search_epsilon": 1e-7,
        "line_search_batch_limit": -1,

    }

    # run NAS search
    nas = NAS(train_loader, valid_loader, metadata, clock=None)
    model = nas.search()

    print(" Final model returned by NAS.search():")
    print(model)

    # training + prediction
    trainer = Trainer(model, epochs=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      train_dataloader=train_loader,
                      valid_dataloader=valid_loader,
                      metadata=metadata,
                      clock=None)
    
    trained_model = trainer.train()
    predictions = trainer.predict(test_loader)

    print("Predictions on test data:", predictions)
 


if __name__ == "__main__":
    main()
  

