import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import math
import torch.nn.functional as F
import numpy as np
import git
import argparse

import sys
if "../" not in sys.path:
    sys.path.append("../")

if '/home/tau/sdouka/codebase/experimental_grow' not in sys.path:
    sys.path.append('/home/tau/sdouka/codebase/experimental_grow')
if "/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/" not in sys.path:
    sys.path.append("/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/")

from tools.datasets import get_dataset
from tools.logger import Logger

def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder

def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="Debug")
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--node_name", type=str)
    args = parser.parse_args()
    return args

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
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, logger, clock):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.logger = logger
        self.clock = clock

        # define training parameters
        self.epochs = metadata['training_epochs']
        self.optimizer = metadata["optimizer"]
        self.scheduler = metadata["scheduler"]
        self.criterion = metadata["criterion"]

        print(f"  Training for {self.epochs} epochs")
        # self.optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=3e-4)
        # self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        # self.criterion = nn.CrossEntropyLoss()
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        # self.scheduler = None

    """
    ====================================================================================================================
    TRAIN ==============================================================================================================
    ====================================================================================================================
    The train function will define how your model is trained on the train_dataloader.
    Output: Your *fully trained* model
    
    See the example submission for how this should look
    """
    def train(self):
        train_accuracies, val_accuracies = [], []
        best_acc = 0.0
        t_start = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            labels, predictions = [], []
            epoch_loss = 0
            gradients = 0
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                # print(f"{data.shape=} {target.shape=}")
                self.optimizer.zero_grad()
                output = self.model(data)

                # store labels and predictions to compute accuracy
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.logger.log_metric("train batch loss", loss.item(), epoch)
                
                try:
                    avg_grad_norm = 0.0
                    for param in self.model.parameters():
                        avg_grad_norm += param.grad.norm().item()
                    avg_grad_norm /= len(list(self.model.parameters()))
                    self.logger.log_metric("gradient batch norm", avg_grad_norm, epoch)
                except AttributeError as err:
                    print(err)
                    for param in self.model.parameters():
                        print(f"{param=} {param.grad}")
                gradients += avg_grad_norm

            if self.scheduler:
                self.scheduler.step()
                    
            self.logger.log_metric("train loss", epoch_loss/len(self.train_dataloader), epoch)
            self.logger.log_metric("gradient norm", gradients/len(self.train_dataloader), epoch)
            train_acc = accuracy_score(labels, predictions)
            valid_acc = self.evaluate()
            self.logger.log_metric("train acc", train_acc, epoch)
            self.logger.log_metric("valid acc", valid_acc, epoch)

            train_accuracies.append(train_acc)
            val_accuracies.append(valid_acc)

            if valid_acc > best_acc:
                best_acc = valid_acc
                # save checkpoint
                torch.save(self.model.state_dict(), "best_allcnn_cifar10.pth")
            
            if (epoch + 1) % 5 == 0 or epoch == self.epochs or self.epochs <= 10:
                print("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} |".format(
                    epoch + 1, self.epochs,
                    train_acc * 100, valid_acc * 100,
                    show_time((time.time() - t_start) / (epoch + 1))
                ))
        print("  Total runtime: {}".format(show_time(time.time() - t_start)))
        return self.model, train_accuracies, val_accuracies

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
        for data, labels in test_loader:
            data = data.to(self.device)
            output = self.model.forward(data)
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions

# Define the ALL-CNN-C model
class AllCNN_C(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(AllCNN_C, self).__init__()
        # According to Springenberg et al. (“Striving for Simplicity”) architecture
        self.dropout0 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)  # downsample 32 → 16
        self.dropout1 = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)  # downsample 16 → 8
        self.dropout2 = nn.Dropout(0.5)
        self.batchnorm2 = nn.BatchNorm2d(192)

        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(192)

        # No fully connected layers; global average pooling will reduce to (num_classes,)
        # Note: The spatial map before GAP should be 8×8, so average over that.

    def forward(self, x):
        x = self.dropout0(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.batchnorm1(self.conv3(x)))  # ← BN after downsampling
        x = self.dropout1(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.batchnorm2(self.conv6(x)))  # ← BN after downsampling
        x = self.dropout2(x)

        x = F.relu(self.batchnorm3(self.conv7(x)))  # ← one BN for stability before final layers
        x = F.relu(self.conv8(x))
        x = self.conv9(x)  # logits, no BN/ReLU

        # Global average pool (spatial → 1×1)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # flatten to (batch, num_classes)
        return x


if __name__ == "__main__":
    args = parse_args()

    trainset, valset, testset = get_dataset("cifar10", "../data", split_train_val=0.3)
    print(f"data shape={trainset.dataset.data.shape[1:]}")
    print(f"train samples={len(trainset)}")
    # in_features = np.prod(trainset.dataset.data.shape[1:])
    out_features = len(np.unique(trainset.dataset.targets))
    # print(f"{in_features=}")
    print(f"{out_features=}")


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

    train_loader = torch.utils.data.DataLoader(trainset, 64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, 64)
    test_loader = torch.utils.data.DataLoader(testset, 128)

    batch_size = 64
    num_epochs = 350
    init_lr = 1e-3
    weight_decay = 1e-3
    momentum = 0.9

    device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AllCNN_C(num_classes=out_features).to(device)

    logger = Logger("Vanilla")
    logger.setup_tracking()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300])

    metadata = {
        "training_epochs": num_epochs,
        "optimizer": optimizer,
        "scheduler": None,
        "criterion": criterion,
        "batch_size": batch_size,
        "lrate": init_lr,
        "weight_decay": weight_decay,
        # "momentum": momentum,
    }

    trainer = Trainer(model, device=device,
            train_dataloader=train_loader,
            valid_dataloader=val_loader,
            metadata=metadata,
            logger=logger,
            clock=None,
    )

    with logger(tags=tags):
        for param, value in metadata.items():
            logger.log_parameter(param, value)

        trained_model, train_acc, val_acc = trainer.train()

    predictions = trainer.predict(test_loader)

    print("Predictions on test data:", predictions)