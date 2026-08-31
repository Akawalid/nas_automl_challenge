from typing import Literal

import numpy as np
import torch
import yaml
from imblearn.under_sampling import RandomUnderSampler  # imblearn -> 0.13.0
from omegaconf import DictConfig
from pyriemann.spatialfilters import Xdawn  # pyriemann n'importe quelle version
from torch.utils.data import TensorDataset

try:
    from tools.datasets import make_dataloader
except ImportError:
    import sys

    sys.path.append("../..")
    print(sys.path)
    from tools.datasets import make_dataloader
try:
    from hydra_script.eeg.Alignments.aligner import Aligner
except ImportError:
    from Alignments.aligner import Aligner
try:
    from hydra_script.eeg.STLDataLoader import STLDataLoader
except ImportError:
    from STLDataLoader import STLDataLoader


__author__ = "Théo Rudkiewicz, Sebastien Velut"


def balance(X, Y, domains):
    X_new = []
    Y_new = []
    domains_new = []
    if domains is not None:
        for d in np.unique(domains):
            ind_domain = np.where(domains == d)
            rus = RandomUnderSampler()
            counter = np.array(range(0, len(Y[ind_domain]))).reshape(-1, 1)
            index, _ = rus.fit_resample(counter, Y[ind_domain])
            index = np.sort(index, axis=0)
            X_new.append(np.squeeze(X[ind_domain][index, :, :], axis=1))
            Y_new.append(np.squeeze(Y[ind_domain][index]))
            domains_new.append(np.squeeze(domains[ind_domain][index]))
        return np.concatenate(X_new), np.concatenate(Y_new), np.concatenate(domains_new)
    else:
        rus = RandomUnderSampler()
        counter = np.array(range(0, len(Y))).reshape(-1, 1)
        index, _ = rus.fit_resample(counter, Y)
        index = np.sort(index, axis=0)
        X = np.squeeze(X[index, :, :], axis=1)
        Y = np.squeeze(Y[index])
        return X, Y, None


def full_preprocessed_data(path, participant):
    return np.load(path + "full_preprocess_data_" + participant + ".npy")


def features_preproc(X, Y, n_cal, n_class=5, window_size=0.35, freqwise=500):
    """
    Preprocess the features before fitting the classifiers

    Parameters:
    X_train: np.array, training data with shape (N_epochs, N_channels, N_samples)

    Y_train: np.array, training labels of the code (0 or 1) with shape (N_epochs,)

    X_test: np.array, testing data with shape (N_epochs, N_channels, N_samples)

    Y_test: np.array, testing labels of the code (0 or 1) with shape (N_epochs,)

    recenter: boolean, Boolean to know if you recenter the data or not

    return: the preprocessed features
    X_train: np.array, preprocessed training data with shape (N_epochs, N_channels, N_samples)

    X_test: np.array, preprocessed testing data with shape (N_epochs, N_channels, N_samples)
    """
    nb_sample_cal = int(n_class * n_cal * (2.2 - window_size) * freqwise)
    X_preproc = np.zeros((X.shape[0], X.shape[1], 8, X.shape[3]))

    for i in range(X.shape[0]):
        X_train = X[i, :nb_sample_cal]
        Y_train = Y[i, :nb_sample_cal]
        X_test = X[i, nb_sample_cal:]

        xdawn = Xdawn(nfilter=4, classes=[1], estimator="lwf")

        X_std = X_train.std(axis=0)
        temp_Xtrain = X_train / (X_std + 1e-8)
        temp_Xtest = X_test / (X_std + 1e-8)
        xdawn = xdawn.fit(temp_Xtrain, Y_train)
        temp_Xtrain = xdawn.transform(temp_Xtrain)
        temp_Xtest = xdawn.transform(temp_Xtest)
        X_preproc[i, :nb_sample_cal] = np.hstack(
            [
                temp_Xtrain,
                np.tile(xdawn.evokeds_[None, :, :], (temp_Xtrain.shape[0], 1, 1)),
            ]
        )
        X_preproc[i, nb_sample_cal:] = np.hstack(
            [temp_Xtest, np.tile(xdawn.evokeds_[None, :, :], (temp_Xtest.shape[0], 1, 1))]
        )

        alig = Aligner(estimator="lwf", metric="real")
        alig = alig.fit(X_preproc[i, :nb_sample_cal])
        X_preproc[i] = alig.transform(X_preproc[i])

    return X_preproc


def get_train_test_data(
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xs: None,
    Ys: None,
    domainst: np.ndarray,
    domainss: None,
    codes: None,
    labels_code: np.ndarray,
    method: Literal["SiSu"],
    clf_name: None,
    n_class: int = 5,
    n_cal: int = 4,
    window_size: float = 0.35,
    freqwise: int = 500,
    test_size: float = 0.2,
):
    if method == "SiSu":
        # Initialisation
        nb_sample_cal = int(n_class * n_cal * (2.2 - window_size) * freqwise)
        nb_sample_val = int(n_class * (n_cal + 1) * (2.2 - window_size) * freqwise)

        # Get the training data
        X_train = Xt[:nb_sample_cal]
        Y_train = Yt[:nb_sample_cal]
        domains_train = domainst[:nb_sample_cal]
        X_val = Xt[nb_sample_cal:nb_sample_val]
        Y_val = Yt[nb_sample_cal:nb_sample_val]
        domains_val = domainst[nb_sample_cal:nb_sample_val]
        X_test = Xt[nb_sample_val:]
        Y_test = Yt[nb_sample_val:]
        domains_test = domainst[nb_sample_val:]
        labels_code_test = labels_code[(n_class * n_cal) :]

        X_train, Y_train, domains_train = balance(X_train, Y_train, domains_train)

    return (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        domains_train,
        domains_val,
        domains_test,
        labels_code_test,
    )


def get_dataset(config: DictConfig, device: torch.device):
    dl = STLDataLoader(
        config["path"],
        config["fmin"],
        config["fmax"],
        config["window_size"],
        config["sample_freq"],
        config["fps"],
        config["timewise"],
        [config["participant"]],
        config["n_class"],
    )
    raw_data = dl.load_data()
    X, Y, domains, codes, labels_code = dl.get_epochs(raw_data)

    Xpreproc = features_preproc(
        X,
        Y,
        n_cal=config["ncal"],  # ask Sebastien about this hardcoded value
        n_class=config["n_class"],
        window_size=config["window_size"],
        freqwise=config["sample_freq"],
    )

    p = 0  # Only one participant
    (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        domains_train,
        domains_val,
        domains_test,
        labels_code_test,
    ) = get_train_test_data(
        np.expand_dims(Xpreproc[p], 1),
        Y[p],
        None,
        None,
        np.array(domains)[p],
        None,
        codes,
        labels_code,
        method="SiSu",
        clf_name="GroCNN",
        n_cal=config["ncal"],
    )

    data = {
        "train": [X_train, Y_train],
        "val": [X_val, Y_val],
        "test": [X_test, Y_test],
    }
    for split in ["train", "val", "test"]:
        assert data[split][0].shape[0] == data[split][1].shape[0], (
            f"Number of {split} samples {data[split][0].shape[0]} does not "
            f"match number of {split} labels {data[split][1].shape[0]}"
        )

        assert (
            data[split][1].ndim == 1
        ), f"Expected {split} labels to be 1-dimensional, got {data[split][1].ndim}-dimensional"
        assert np.unique(data[split][1]).shape[0] == 2, (
            f"Number of unique classes in {split} labels "
            f"{np.unique(data[split][1]).shape[0]} does not match expected "
            f"number of classes {config['n_class']}"
        )

        print(
            f"X_{split} shape: {data[split][0].shape}, "
            f"Y_{split} shape: {data[split][1].shape}, "
            f"Unique classes in Y_{split}: {np.unique(data[split][1])}"
        )

    # NOTE: Tensors must be created on CPU for DataLoader with num_workers > 0.
    # CUDA tensors in the dataset cause "CUDA initialization error" when workers
    # try to access them after forking. Data is moved to GPU in the training loop.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(
        Y_train,
        dtype=int,
    )
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(
        Y_val,
        dtype=int,
    )
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(
        Y_test,
        dtype=int,
    )

    # Create DataLoader for train, validation, and test sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, val_dataset, test_dataset


def get_dataloader_eeg(
    config: DictConfig,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 0,
):
    train_dataset, val_dataset, test_dataset = get_dataset(config, device)

    default_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "device": device,
        "seed": seed,
    }
    options = {
        "train": {k: v for k, v in default_options.items()},
        "val": {k: v for k, v in default_options.items()},
        "test": {k: v for k, v in default_options.items()},
    }
    options["train"]["shuffle"] = True
    options["train"]["drop_last"] = True

    train_loader = make_dataloader(train_dataset, **options["train"])
    val_loader = make_dataloader(val_dataset, **options["val"])
    test_loader = make_dataloader(test_dataset, **options["test"])

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import os

    print(f"Current working directory: {os.getcwd()}")
    # Example usage
    config_path = "hydra_script/configs/dataset_config/dry-ricker.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    train_dataset, val_dataset, test_dataset = get_dataset(config, device)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print("Datasets loaded successfully.")
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloader_eeg(
        config, device, batch_size=64, num_workers=4
    )
    print("DataLoaders created successfully.")
    print(f"Train DataLoader size: {len(train_loader)}")
    print(f"Validation DataLoader size: {len(val_loader)}")
    print(f"Test DataLoader size: {len(test_loader)}")
    print("DataLoaders are ready for use.")
