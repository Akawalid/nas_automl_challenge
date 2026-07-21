import torch
from torch.utils import data
from torchvision.transforms import v2 as Tv2  # this is the new system
import numpy as np
import os

from gromo.utils.utils import global_device
from helpers import get_transforms

# custom dataset class for Pytorch and transform applications

class TorchDataset(data.Dataset):
                  
    def __init__(self, data, labels=None, transform=None):
        self.data = torch.from_numpy(data).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        if transform:
            self.data = [transform(x) for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # (C, H, W)

        if self.labels is not None:
            return x, self.labels[idx] 
        else:
            return x
        

class DataProcessor:
    """
    -===================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs:
        * train_x: numpy array of shape [n_train_datapoints, channels, height, width], these are the training inputs
        * train_y: numpy array of shape [n_train_datapoints], these are the training labels
        * valid_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the validation inputs
        * valid_y: numpy array of shape [n_valid_datapoints], these are the validation labels
        * test_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the test inputs
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission

    You can modify or add anything into the metadata that you wish, if you want to pass messages between your classes

    """

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata):
        self.train_x =  train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        pass


    """
    ====================================================================================================================
    PROCESS ============================================================================================================
    ====================================================================================================================
    This function will be called, and it expects you to return three outputs:
        * train_loader: A Pytorch dataloader of (input, label) tuples
        * valid_loader: A Pytorch dataloader of (input, label) tuples
        * test_loader: A Pytorch dataloader of (inputs)  <- Make sure shuffle=False and drop_last=False!
        
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for more info.  
        
    Here, you can do whatever you want to the input data to process it for your NAS algorithm and training functions
    """

    # ====================================================================================================================
    # still need to impletment filter_classes, find_data_and_labels, load_data, get_item, get_num_classes
    # ====================================================================================================================

    def process(self):

        # get the transforms required for the metadata 
        # data_augmentation = ['horizontal_flip', 'rotation', 'crop', 'autoaugment', 'randaugment'] 
        base_transforms, augmentation_transforms = get_transforms(
            self.metadata['codename'])
        train_tf = Tv2.Compose(augmentation_transforms + base_transforms)
        valid_tf = Tv2.Compose(base_transforms)
        
        # print("Sample image shape:", self.train_x[0].shape, "dtype:", self.train_x[0].dtype)

        #to_pil = ToPILImage()

        # convert numpy arrays (HWC form) → PIL images → apply transforms (augmentaiton, to tensor, normalization)
        #train_tensor = [train_tf(to_pil(x)) for x in self.train_x]
        #valid_tensor = [valid_tf(to_pil(x)) for x in self.valid_x]
        #test_tensor  = [valid_tf(to_pil(x)) for x in self.test_x]

        #train_x = torch.stack(train_tensor)
        #valid_x = torch.stack(valid_tensor)
        #test_x  = torch.stack(test_tensor)

        # wrap the datasets 
        #train_dataset = torch.utils.data.TensorDataset(
        #   train_x, torch.tensor(self.train_y, dtype=torch.long))
        #valid_dataset = torch.utils.data.TensorDataset(
        #    valid_x, torch.tensor(self.valid_y, dtype=torch.long))
        #test_dataset = torch.utils.data.TensorDataset(
        #    test_x)

        train_dataset = TorchDataset(self.train_x, self.train_y, transform=train_tf)
        valid_dataset = TorchDataset(self.valid_x, self.valid_y, transform=valid_tf)
        test_dataset = TorchDataset(self.test_x, transform=valid_tf)
        
        # create dataloaders and set parameters 
        batch_size: int = 1024
        num_workers: int = 0
        device: torch.device = global_device()
        shuffle: bool = True

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False)  

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)   

        
        return train_loader, valid_loader, test_loader

def load_cifar10_numpy():
        import torchvision
        from sklearn.model_selection import train_test_split

        # Download training set
        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        train_x = np.array(cifar10_train.data, dtype=np.uint8)  # shape: (50000, 32, 32, 3)
        train_y = np.array(cifar10_train.targets, dtype=np.int64)  # shape: (50000,)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)
        
        # Download validation/test set
        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        
        test_x = np.array(cifar10_test.data, dtype=np.uint8)
        test_y = np.array(cifar10_test.targets, dtype=np.uint8)

        np.save("data/cifar-10/train_x.npy", train_x)
        np.save("data/cifar-10/train_y.npy", train_y)
        np.save("data/cifar-10/valid_x.npy", valid_x)
        np.save("data/cifar-10/valid_y.npy", valid_y)
        np.save("data/cifar-10/test_x.npy", test_x)
        np.save("data/cifar-10/test_y.npy", test_y)

        # return train_x, train_y, valid_x, valid_y, test_x, test_y


if __name__ == "__main__":
    import numpy as np

    metadata = {
        'num_classes': 10,
        'codename': 'cifar10',  
        'input_shape': (100, 32, 32, 3),  # HWC
        'time_remaining': 3600
    }

    # NumPy arrays in HWC format 
    train_x = np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8)
    train_y = np.random.randint(0, 10, size=(100,))
    valid_x = np.random.randint(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
    valid_y = np.random.randint(0, 10, size=(20,))
    test_x  = np.random.randint(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)

    # run processor
    processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata)
    train_loader, valid_loader, test_loader = processor.process()

    # check shape of one batch
    print("Train batch shape:", next(iter(train_loader))[0].shape)
    print("Valid batch shape:", next(iter(valid_loader))[0].shape)
    print("Test batch shape:", next(iter(test_loader))[0].shape)

    import json
    with open(os.path.join("datasets/cifar-10", 'metadata'), "r") as f:
        metadata = json.load(f)
    print(metadata)
