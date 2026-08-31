from pyriemann.utils.mean import mean_covariance
from sklearn.base import BaseEstimator, TransformerMixin

from .covariance import compute_covariances
from .riemannian import compute_riemannian_alignment


class Aligner(BaseEstimator, TransformerMixin):

    def __init__(self, estimator="lwf", metric="real"):
        """
        Initialisation
        """
        self.estimator = estimator
        self.metric = metric

    def fit(self, X, Y=None):
        """Fit.

        Estimate alignment response for current subject.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times) if 'real' matrix; shape (n_matrices, n_channels, n_channels) if "riemann" metric
            Multi-channel time-series.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix. not used here

        Returns
        -------
        self : Aligner instance
            The Aligner instance.
        """

        if self.metric == "riemann":
            self.rmean = mean_covariance(X, metric="riemann")
        else:
            self.rmean = mean_covariance(
                compute_covariances(X, estimator="lwf"), metric="riemann"
            )

        return self

    def transform(self, X):
        """Estimate Aligned matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times) if 'real' matrix; shape (n_matrices, n_channels, n_channels) if "riemann" metric
            Multi-channel time-series.

        Returns
        -------
        align_mats : ndarray, shape (n_matrices, n_channels, n_times) if 'real' matrix; shape (n_matrices, n_channels, n_channels) if "riemann" metric
            Aligned matrices. The alignment was made in the Riemann
            or in the Euclidian space but with the fitted riemannian mean.
        """
        align_mats = compute_riemannian_alignment(X, mean=self.rmean, dtype=self.metric)
        return align_mats

    def fit_transform(self, X, Y=None):
        """Fit alignment response for current subject and Estimate Aligned matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix. not used here

        Returns
        -------
        align_mats : ndarray, shape (n_matrices, n_channels, n_times) if 'real' matrix; shape (n_matrices, n_channels, n_channels) if "riemann" metric
            Aligned matrices. The alignment was made in the Riemann
            or in the Euclidian space but with the fitted riemannian mean.
        """

        if self.metric == "riemann":
            self.rmean = mean_covariance(X, metric="riemann")
        else:
            self.rmean = mean_covariance(
                compute_covariances(X, estimator="lwf"), metric="riemann"
            )

        align_mats = compute_riemannian_alignment(X, mean=self.rmean, dtype=self.metric)
        return align_mats
