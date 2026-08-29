import torch
from torch.utils import data
from torchvision.transforms import v2 as Tv2

from helpers import get_transforms


class TorchDataset(data.Dataset):
    def __init__(self, data, labels=None, transform=None):
        # Raw arrays are already (N, C, H, W) -- verified against the actual competition data's
        # raw .npy headers (e.g. Chesseract's train_x.npy is (49998, 12, 8, 8): 12 chess-encoding
        # planes, 8x8 board -- unambiguously channel-first, not channel-last). No permute needed;
        # a channel-last->channel-first permute here would silently scramble width into the
        # channel axis and channels into the height axis instead. See nas.py's matching fix.
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        if transform:
            self.data = [transform(x) for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.labels is not None:
            return x, self.labels[idx]
        return x


class DataProcessor:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs:
        * train_x/valid_x/test_x: numpy arrays of shape [n, channel, height, width]
        * train_y/valid_y: numpy arrays of shape [n]
        * metadata: dict with 'num_classes', 'codename', 'input_shape', 'time_remaining'
    """

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock=None):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    def process(self):
        base_transforms = get_transforms(self.metadata['codename'], sample_data=self.train_x)
        tf = Tv2.Compose(base_transforms)

        train_dataset = TorchDataset(self.train_x, self.train_y, transform=tf)
        valid_dataset = TorchDataset(self.valid_x, self.valid_y, transform=tf)
        test_dataset = TorchDataset(self.test_x, transform=tf)

        batch_size = 256  # Appendix B training-phase batch size

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader
