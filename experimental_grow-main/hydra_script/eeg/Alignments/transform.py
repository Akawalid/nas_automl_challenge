import copy

import numpy as np
from braindecode.datasets import create_from_X_y
from sklearn.base import BaseEstimator, TransformerMixin

from .chordal import chordal_alignment, compute_chordal
from .covariance import compute_covariances, mean_group
from .euclidean import euclidean_alignment
from .geodesic import compute_geodesic, geodesic_alignment
from .kullback import compute_kullback, kl_alignment
from .one_class_alignment import compute_ref_oneclass
from .riemannian import (
    compute_resting_alignment,
    riemannian_alignment,
    riemannian_resting_alignment,
)


# Does nothing if X is array. Returns X[0] is list
# Just a requirement of my pipeline
class DoNotTransform(BaseEstimator, TransformerMixin):

    def __init__(self, list_mean_r=None, list_mean_e=None, kw_args=None):
        self.kw_args = kw_args
        self.list_mean_r = list_mean_r
        self.list_mean_e = list_mean_e

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        """If the X is ndarray, does nothing. If it's a list with first element an array, returns X[0].

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
            Or
            X : list, shape (2,)
            X[0] is ndarray, shape (n_trials, n_channels, n_channels)
            X[1] is ndarray, shape (n_trials,)

        """
        if isinstance(X, list):
            X = X[0]
        return X

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


# Euclidean Alignment transformation (He He, Dongrui Wu 2018)
# Other functions implemented in euclidean.py
class TransformEA(BaseEstimator, TransformerMixin):

    def __init__(self, size=None, estimator="lwf", kw_args=None):
        self.kw_args = kw_args
        # Size of the set to be aligned
        self.size = size
        self.estimator = estimator
        self.domain = None

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        """Align X array based on self.size (if X is array) or X[1] (if X is list).
        Always returns the same type as X inputted

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
            Or
            X : list, shape (2,)
            X[0] is ndarray, shape (n_trials, n_channels, n_channels)
            X[1] is ndarray, shape (n_trials,)

        """
        if isinstance(X, list):
            domain = X[1]
            self.domain = domain
            X = X[0]

        align = euclidean_alignment(
            X, size=self.size, domain=self.domain, estimator=self.estimator
        )

        return align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


# Recentering step of the RPA (Pedro Rodrigues, 2018)
class TransformRR(BaseEstimator, TransformerMixin):

    def __init__(self, size=None, kw_args=None):
        self.kw_args = kw_args
        self.size = size
        self.domain = None

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        if isinstance(X, list):
            domain = X[1]
            self.domain = domain
            X = X[0]

        align = riemannian_alignment(X, size=self.size, domain=self.domain)

        return align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


# Geodesic combination
class TransformGeodesic(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        alpha=0.5,
        dtype="covmat",
        ref=None,
        list_mean_r=None,
        list_mean_e=None,
        size=None,
    ):

        self.size = size
        self.ref = ref
        self.list_mean_r = list_mean_r
        self.list_mean_e = list_mean_e
        self.alpha = alpha
        self.dtype = dtype

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):

        X = copy.deepcopy(X)

        if isinstance(X, np.ndarray):
            # "online evaluation" mode
            if self.ref is not None:
                align = compute_geodesic(
                    X, alpha=self.alpha, dtype=self.dtype, r_hre=self.ref
                )

            # "Offline evaluation" mode
            else:

                # Compute cov matrices if needed
                if self.dtype == "covmat":
                    cov = X
                else:
                    cov = compute_covariances(X, estimator="lwf")

                if self.size is None:
                    self.size = cov.shape[0]

                # Compute mean cov matrices per group
                if self.list_mean_r is None:
                    list_mean_r = mean_group(cov, size=self.size, metric="riemann")
                else:
                    list_mean_r = self.list_mean_r

                if self.list_mean_e is None:
                    list_mean_e = mean_group(cov, size=self.size, metric="euclid")
                else:
                    list_mean_e = self.list_mean_e

                align = geodesic_alignment(
                    X, list_mean_r, list_mean_e, alpha=self.alpha, dtype=self.dtype
                )
        else:
            domains = X[1]
            X = X[0]

            # "online evaluation" mode
            if self.ref is not None:
                align = compute_geodesic(
                    X, alpha=self.alpha, dtype=self.dtype, r_hre=self.ref
                )

            else:
                # Compute cov matrices if needed
                if self.dtype == "covmat":
                    cov = X
                else:
                    cov = compute_covariances(X, estimator="lwf")

                # Compute mean cov matrices per group
                if self.list_mean_r is None:
                    list_mean_r = mean_group(cov, domain=domains, metric="riemann")
                else:
                    list_mean_r = self.list_mean_r

                if self.list_mean_e is None:
                    list_mean_e = mean_group(cov, domain=domains, metric="euclid")
                else:
                    list_mean_e = self.list_mean_e

                align = geodesic_alignment(
                    X,
                    list_mean_r,
                    list_mean_e,
                    alpha=self.alpha,
                    dtype=self.dtype,
                    domains=domains,
                )

        return align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


# Chordal combination
class TransformChordal(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        alpha=0.5,
        dtype="covmat",
        ref=None,
        list_mean_r=None,
        list_mean_e=None,
        size=None,
    ):

        self.size = size
        self.ref = ref
        self.list_mean_r = list_mean_r
        self.list_mean_e = list_mean_e
        self.alpha = alpha
        self.dtype = dtype

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):

        X = copy.deepcopy(X)

        if isinstance(X, np.ndarray):
            # "online evaluation" mode
            if self.ref is not None:
                align = compute_chordal(
                    X, alpha=self.alpha, dtype=self.dtype, r_hre=self.ref
                )

            # "Offline evaluation" mode
            else:

                # Compute cov matrices if needed
                if self.dtype == "covmat":
                    cov = X
                else:
                    cov = compute_covariances(X, estimator="lwf")

                if self.size is None:
                    self.size = cov.shape[0]

                # Compute mean cov matrices per group
                if self.list_mean_r is None:
                    list_mean_r = mean_group(cov, size=self.size, metric="riemann")
                else:
                    list_mean_r = self.list_mean_r

                if self.list_mean_e is None:
                    list_mean_e = mean_group(cov, size=self.size, metric="euclid")
                else:
                    list_mean_e = self.list_mean_e

                align = chordal_alignment(
                    X,
                    list_mean_r,
                    list_mean_e,
                    alpha=self.alpha,
                    dtype=self.dtype,
                )
        else:
            domains = X[1]
            X = X[0]

            # "online evaluation" mode
            if self.ref is not None:
                align = compute_chordal(
                    X, alpha=self.alpha, dtype=self.dtype, r_hre=self.ref
                )

            else:
                # Compute cov matrices if needed
                if self.dtype == "covmat":
                    cov = X
                else:
                    cov = compute_covariances(X, estimator="lwf")

                # Compute mean cov matrices per group
                if self.list_mean_r is None:
                    list_mean_r = mean_group(cov, domain=domains, metric="riemann")
                else:
                    list_mean_r = self.list_mean_r

                if self.list_mean_e is None:
                    list_mean_e = mean_group(cov, domain=domains, metric="euclid")
                else:
                    list_mean_e = self.list_mean_e

                align = chordal_alignment(
                    X,
                    list_mean_r,
                    list_mean_e,
                    alpha=self.alpha,
                    dtype=self.dtype,
                    domains=domains,
                )

        return align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


class TransformToWindows(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=None, kw_args=None):
        self.kw_args = kw_args
        self.sfreq = sfreq

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):

        if y is None:
            y = self.y

        if isinstance(X, np.ndarray) or isinstance(X, list):

            if isinstance(X, list):
                X = X[0]

            dataset = create_from_X_y(
                X=X,
                y=y,
                window_size_samples=X.shape[2],
                window_stride_samples=X.shape[2],
                drop_last_window=False,
                sfreq=self.sfreq,  # Think later
            )

        else:

            dataset = create_from_X_y(
                X=X.get_data(),
                y=y,
                window_size_samples=X.get_data().shape[2],
                window_stride_samples=X.get_data().shape[2],
                drop_last_window=False,
                sfreq=X.info["sfreq"],
            )

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


# Zanini et al., 2019
# Uses data from resting state to align
class TransformRRA(BaseEstimator, TransformerMixin):

    def __init__(self, t_break, size=None, estimator="lwf", r_ref=None, dtype="covmat"):
        self.dtype = dtype
        self.size = size
        self.estimator = estimator
        # Struggling a little on how/where to define it
        self.t_break = t_break
        self.r_ref = r_ref

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        assert isinstance(X, np.ndarray)

        t_end = self.t_break

        X_rest = X[:, :, t_end:]
        X = X[:, :, :t_end]

        # Compute the covariances
        if self.dtype == "covmat":
            X = compute_covariances(X, estimator=self.estimator)
        X_rest = compute_covariances(X_rest, estimator=self.estimator)

        if self.size is None:
            self.size = X.shape[0]

        # Training/Offline mode
        if self.r_ref is None:
            align = riemannian_resting_alignment(
                X, X_rest, size=self.size, dtype=self.dtype
            )
        # Online mode
        else:
            align = compute_resting_alignment(
                X, X_rest, r_ra=self.r_ref, dtype=self.dtype
            )

        return align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


# Use Kullback-Leibler divergence to compute mean
class TransformKL(BaseEstimator, TransformerMixin):

    def __init__(self, dtype="covmat", ref=None, size=None, sample_weight=None):

        self.size = size
        self.ref = ref
        self.sample_weight = sample_weight
        self.dtype = dtype

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):

        if isinstance(X, np.ndarray):
            # "online evaluation" mode
            if self.ref is not None:
                align = compute_kullback(
                    X, dtype=self.dtype, r_kl=self.ref, sample_weight=self.sample_weight
                )

            # "Offline evaluation" mode
            else:

                align = kl_alignment(
                    X, sample_weight=self.sample_weight, size=self.size, dtype=self.dtype
                )
        else:
            domains = X[1]
            X = X[0]

            # "online evaluation" mode
            if self.ref is not None:
                align = compute_kullback(X, dtype=self.dtype, r_kl=self.ref)

            else:

                align = kl_alignment(
                    X, sample_weight=self.sample_weight, domains=domains, dtype=self.dtype
                )

        return align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


class TransformOneClass(BaseEstimator, TransformerMixin):

    def __init__(self, reflabel=0, dtype="covmat"):
        self._ref = list()
        self.reflabel = reflabel
        self.dtype = dtype

    def _compute_ref(self, X):
        ref = compute_ref_oneclass(X, dtype=self.dtype)
        self._ref.append(ref)

    def _fit_oca_sw(self, X, domains, y):

        for d in np.unique(domains):
            X_d = X[domains == d]
            y_d = y[domains == d]

            X_ref = X_d[y_d == self.reflabel]
            self._compute_ref(X_ref)

    def _transform_oca_sw(self, X, domains):
        X_align = []
        for i in range(len(np.unique(domains))):
            d = np.unique(domains)[i]
            X_d = X[domains == d]
            ref_d = self._ref[i]

            if self.dtype == "covmat":
                align = ref_d @ X_d @ ref_d
            else:
                align = ref_d @ X_d
            X_align.append(align)

        X_align = np.concatenate(X_align)
        return X_align

    def fit(self, X, y=None):

        self._ref.clear()

        if isinstance(X, list):
            domains = X[1]
            X = X[0]
            self._fit_oca_sw(X, domains, y)

        else:
            X_ref = X[y == self.reflabel]
            self._compute_ref(X_ref)

        return self

    def transform(self, X, y=None):

        if isinstance(X, list):
            domains = X[1]
            X = X[0]
            X_align = self._transform_oca_sw(X, domains)
        else:
            ref = self._ref[0]
            if self.dtype == "covmat":
                X_align = ref @ X @ ref
            else:
                X_align = ref @ X

        return X_align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True
