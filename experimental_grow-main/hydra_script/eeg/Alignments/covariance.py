import copy

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import *
from pyriemann.utils.mean import mean_covariance
from sklearn.base import BaseEstimator, TransformerMixin


def compute_covariances(data, estimator="lwf"):
    data = copy.deepcopy(data)

    assert len(data.shape) == 3

    est = Covariances(estimator=estimator)

    covmats = est.transform(data)

    return covmats


def mean_group(covmats, size=24, domain=None, metric="riemann"):
    # Returns a list containing the mean of each batch/group of matrices
    covmats_means = []
    if domain is not None:
        for s in np.unique(domain):
            cov_batch = covmats[domain == s]
            cov_batch_mean = mean_covariance(cov_batch, metric=metric)
            covmats_means.append(cov_batch_mean)
    else:
        m = size
        n = covmats.shape[0]
        for k in range(int(n / m)):
            cov_batch = covmats[k * m : (k + 1) * m]
            cov_batch_mean = mean_covariance(cov_batch, metric=metric)
            covmats_means.append(cov_batch_mean)
    return covmats_means


# Compute mean covariance matrix for each class of each subject
def mean_class(cov, y, refclass=None, metric="riemann"):

    labels = np.unique(y)

    if refclass is None:

        means = []

        for label in labels:
            cov_label = cov[y == label]
            mean_label = mean_covariance(cov_label, metric=metric)

            means.append(mean_label)

    else:
        cov_label = cov[y == refclass]
        means = mean_covariance(cov_label, metric=metric)

    return means


def means_subjects(cov, y, subjects, refclass=None, metric="riemann"):
    labels = np.unique(y)

    means = []

    for subj in np.unique(subjects):
        cov_subj = cov[subjects == subj]
        y_subj = y[subjects == subj]

        means_subj = mean_class(cov_subj, y_subj, metric=metric, refclass=refclass)
        means.append(means_subj)

    return means, labels


def means_group(cov, y, meta, metric="riemann", session=False, run=False):
    subjects = meta.subject.values
    runs = meta.run.values
    sessions = meta.session.values

    if run:
        means = []

        for s in np.unique(sessions):
            mean_runs = []

            ses = sessions == s

            for r in np.unique(runs):
                k = runs == r

                aux_run = np.logical_and(ses, k)

                mean_group, labels = means_subjects(
                    cov[aux_run], y[aux_run], subjects[aux_run], metric=metric
                )
                mean_runs.append(mean_group)

            means.append(mean_runs)

    elif session and not run:
        means = []

        for s in np.unique(sessions):
            ses = sessions == s

            mean_groups, labels = means_subjects(
                cov[ses], y[ses], subjects[ses], metric=metric
            )
            means.append(mean_groups)

    else:
        means, labels = means_subjects(cov, y, subjects, metric=metric)

    return means, labels


# Takes raw as input and returns cov matrices
class TransformCov(BaseEstimator, TransformerMixin):

    def __init__(self, estimator="lwf", kw_args=None):
        self.kw_args = kw_args
        self.estimator = estimator

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):

        if isinstance(X, np.ndarray):
            cov = compute_covariances(X, estimator=self.estimator)
        else:
            domains = X[1]
            X = X[0]
            cov = compute_covariances(X, estimator=self.estimator)

            cov = [cov, domains]

        return cov

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True
