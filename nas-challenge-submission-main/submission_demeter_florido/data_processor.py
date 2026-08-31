import os
import sys

# Make the vendored gromo/experiments/tools packages importable regardless of cwd.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_packages"))

import numpy as np
import torch
import torchvision.transforms as _tvt
from torch.utils.data import DataLoader, Dataset, random_split

from tools.augmentations import default_augmentations, get_transforms

# ====================================================================================================================
# This submission (unlike submission_demeter) does NOT reimplement Demeter's algorithm from the
# paper's pseudocode. It vendors Santiago Florido's own real code (experimental_grow-main's
# experiments/pipeline/pipeline.py, tools/datasets.py, tools/augmentations.py, etc., copied
# verbatim into python_packages/ -- see nas.py for how NAS.search() drives them) and only adds the
# minimum glue needed to fit the challenge's DataProcessor/NAS/Trainer contract: this file replaces
# pipeline.py's own `load_data` step (which downloads a dataset from a path) with a version that
# builds the same kind of Dataset objects directly from the numpy arrays the challenge hands us.
# ====================================================================================================================

# codename (as printed in the challenge's own metadata) -> the real dataset name Florido's code
# uses for augmentation/transform lookup (tools/augmentations.py's default_augmentations/
# npy_datasets dicts are keyed by these lowercase names, not the challenge's arbitrary codenames).
CODENAME_TO_DATASET_NAME = {
    "Mateo": "multnist",
    "Caitie": "cifartile",
    "Gutenberg": "gutenberg",
    "Sadie": "geoclassing",
    "Chester": "chesseract",
}


class PipelineStyleDataset(Dataset):
    """
    Mirrors tools.datasets.NpyWebDataset's in-memory interface (.data in (N,H,W,C), .targets,
    transform-applying __getitem__) but built directly from already-loaded numpy arrays instead of
    downloading+extracting a zip -- the challenge hands us numpy arrays directly, no path to read
    from. NpyWebDataset._load_data() does this same permute on its own raw (N,C,H,W) npy files
    ("Reshape to (N, H, W, C)... matching the torchvision dataset convention" -- its own comment),
    replicated verbatim here since the challenge's raw arrays are the same (N,C,H,W) layout
    (confirmed directly against the real competition data's .npy headers).
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        self.data = torch.from_numpy(data).permute(0, 2, 3, 1).contiguous()
        self.targets = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index].numpy().copy()
        y = self.targets[index].item()
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class UnlabeledDataset(Dataset):
    """
    Plain dataset for the challenge's own test_loader. Test labels are never available to the
    submission at all (evaluation/main.py never passes test_y in), and pipeline.py's own internal
    algorithm never touches this specific split either -- its own "test_set" concept (see
    tools.datasets.get_dataset) comes from the public dataset's separate test_x/test_y files, used
    only for the paper's own final reported numbers, not by NAS.search() itself. Kept deliberately
    simple rather than routed through PipelineStyleDataset.
    """

    def __init__(self, data: np.ndarray, transform=None):
        self.data = torch.from_numpy(data).permute(0, 2, 3, 1).contiguous()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index].numpy().copy()
        if self.transform is not None:
            x = self.transform(x)
        return x


class DataProcessor:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs
        * train_x/valid_x/test_x: numpy arrays of shape [n, channel, height, width]
        * train_y/valid_y: numpy arrays of shape [n]
        * metadata: dict with 'num_classes', 'codename', 'input_shape', 'time_remaining'
    """

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock=None):
        self.train_x, self.train_y = train_x, train_y
        self.valid_x, self.valid_y = valid_x, valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    def process(self):
        dataset_name = CODENAME_TO_DATASET_NAME.get(self.metadata.get("codename"))
        if dataset_name is not None:
            base_transforms, aug_transforms = get_transforms(
                dataset_name, default_augmentations[dataset_name]
            )
        else:
            # Real (hidden) competition dataset we don't have a name for -- no augmentation,
            # matching get_transforms's own generic npy fallback (ToTensor only).
            base_transforms, aug_transforms = [_tvt.ToTensor()], []
            dataset_name = "unknown"
        self.metadata["_pipeline_dataset_name"] = dataset_name

        # npy datasets: augment AFTER ToTensor, on tensors -- matches get_transforms's documented
        # ordering and pipeline.py's own load_data (`if dataset_name in npy_datasets: full_aug =
        # datasets_transforms + augmentation_transforms`); all 5 known datasets here are npy ones.
        train_transform = _tvt.Compose(base_transforms + aug_transforms)
        base_transform = _tvt.Compose(base_transforms)

        # Combine train+valid into ONE pool -- matches tools.datasets.NpyWebDataset(train=True),
        # which loads BOTH the dataset's "train" and "valid" npy files together (see
        # _find_data_and_labels: `prefix = ["train", "valid"] if self.train else ["test"]`). The
        # figshare "valid" split is NOT held out separately by the real pipeline -- it's merged
        # into the combined pool, which tools.datasets.get_dataset then re-splits itself (see
        # below). This is what actually produced the paper's own reported numbers, so it's
        # replicated here rather than treating the challenge's own valid_x as a fixed split.
        combined_x = np.concatenate([self.train_x, self.valid_x], axis=0)
        combined_y = np.concatenate([self.train_y, self.valid_y], axis=0)

        # get_dataset() builds TWO separate dataset instances over the SAME combined pool -- one
        # with the augmenting transform (used for the train split), one without (used for the val
        # split) -- then does ONE random_split and reassigns each split's `.dataset` to whichever
        # instance matches its role (tools/datasets.py's get_dataset, ~lines 786-839). Replicated
        # verbatim rather than approximated.
        train_val_with_aug = PipelineStyleDataset(combined_x, combined_y, transform=train_transform)
        train_val_plain = PipelineStyleDataset(combined_x, combined_y, transform=base_transform)

        val_split = self.metadata.get("val_split", 0.1)
        n_val = max(1, int(round(len(train_val_plain) * val_split)))
        n_train = len(train_val_plain) - n_val
        train_split, val_split_ds = random_split(train_val_plain, [n_train, n_val])
        train_split.dataset = train_val_with_aug
        val_split_ds.dataset = train_val_plain

        test_dataset = UnlabeledDataset(self.test_x, transform=base_transform)

        # pipeline.py's own create_dataloaders step requires a test_set positional argument too
        # (get_dataset's real test_set comes from the dataset's own labeled test_x/test_y files),
        # but the challenge never gives this submission real test labels at all (evaluation/
        # main.py never passes test_y in) -- nothing in the growth loop pipeline.py actually runs
        # here touches test_dataloader (test data is for the paper's own FINAL reported numbers,
        # not for any growth decision), so a placeholder all-zero label vector is enough to
        # satisfy the interface without affecting anything that matters.
        placeholder_test_labels = np.zeros(len(self.test_x), dtype=np.int64)
        pipeline_test_set = PipelineStyleDataset(self.test_x, placeholder_test_labels, transform=base_transform)

        # Stash the split Dataset objects for NAS.search() to use directly. pipeline.py's own
        # split_data/create_dataloaders steps need these *unsplit-again* train_set/val_set/test_set
        # objects (they do their own further growth_set/dev_set split on top every growth round),
        # not flat DataLoaders -- metadata is a plain dict passed by reference through main.py's
        # whole calling chain (DataProcessor -> NAS -> Trainer all receive the SAME object), and
        # the challenge's own DataProcessor docstring explicitly sanctions this pattern: "You can
        # modify or add anything into the metadata that you wish, if you want to pass messages
        # between your classes."
        self.metadata["_pipeline_train_set"] = train_split
        self.metadata["_pipeline_val_set"] = val_split_ds
        self.metadata["_pipeline_test_set"] = pipeline_test_set

        batch_size = self.metadata.get("train_batch_size", 256)
        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(val_split_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, valid_loader, test_loader
