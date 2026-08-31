import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from gromo.utils.utils import global_device
from helpers.auxilliary_functions import *
from pyriemann.spatialfilters import Xdawn

sys.path.append("/mnt/beegfs/home/velut/Seb/Green/Scripts/")
from Alignments.aligner import Aligner
from logger import Logger
from utils import balance

__author__ = "Sebastien Velut"

torch.set_default_device(global_device())


def setup_logger(cfg, run_name):
    api = cfg["logger"]["api"]
    exp_name = cfg["logger"]["exp_name"]
    port = cfg["logger"]["port"]
    enabled = cfg["logger"]["enabled"]
    path = cfg["logger"]["path"]
    logger = Logger(
        experiment_name=exp_name, port=port, api=api, enabled=enabled, run_name=run_name
    )
    logger.setup_tracking(file_path=path)
    return logger


def setup_experiment_tags(cfg):
    try:
        gpu_index = torch.cuda.current_device()
    except:
        gpu_index = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--node_name", type=str)
    args, unknown = parser.parse_known_args()
    tags = {
        "slurm.job_id": args.job_id,
        "slurm.node_name": args.node_name,
        "gpu_index": gpu_index,
    }
    return tags


class AccuracyXP(nn.Module):
    def __init__(self, k: int = 1, reduction: str = "sum"):
        super(AccuracyXP, self).__init__()
        assert reduction in [
            "mean",
            "sum",
            "none",
        ], "reduction should be in ['mean', 'sum', 'none']"
        self.reduction = reduction
        self.k = k

    def forward(self, y_pred, y):
        result = y_pred.topk(self.k, dim=1).indices == y[:, 1].unsqueeze(1)
        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return result.mean()
        elif self.reduction == "sum":
            return result.sum()
        else:
            raise ValueError("reduction should be in ['mean', 'sum', 'none']")


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


def freeze_model(model):
    ct = 0
    for param in model.parameters():
        ct += 1
        if ct < 13:
            param.requires_grad = False


def is_growth_epoch(step: int, epochs_per_growth) -> bool:
    assert step > 0, "Step should be greater than 0"
    if epochs_per_growth == -1:
        return False
    else:
        return step % (epochs_per_growth + 1) == 0


def get_train_test_data(
    Xt,
    Yt,
    Xs,
    Ys,
    domainst,
    domainss,
    codes,
    labels_code,
    method,
    clf_name,
    n_class=5,
    n_cal=4,
    window_size=0.35,
    freqwise=500,
    test_size=0.2,
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
