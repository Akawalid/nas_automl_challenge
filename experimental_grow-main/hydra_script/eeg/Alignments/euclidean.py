import copy

import numpy as np
from numpy import any, iscomplexobj, isfinite, real
from scipy.linalg import inv, sqrtm

from .covariance import compute_covariances, mean_covariance


# Return Euclidean Alignment's reference matrix
def compute_ref_euclidean(data=None, mean=None, dtype="raw", estimator="lwf"):
    if dtype != "raw":
        mean = mean_covariance(data, metric="euclid")

    if mean is None:
        covmats = compute_covariances(data, estimator=estimator)

        mean = mean_covariance(covmats, metric="euclid")

    compare = np.allclose(mean, np.identity(mean.shape[0]))

    if not compare:

        if iscomplexobj(mean):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(mean)):
            print("covariance matrix problem sqrt")

        r_ea = inv(sqrtm(mean))

        if iscomplexobj(r_ea):
            print(
                "WARNING! Covariance matrix was not SPD somehow. "
                + "Can be caused by running ICA-EOG rejection, if "
                + "not, check data!!"
            )
            r_ea = real(r_ea).astype(np.float64)
        elif not any(isfinite(r_ea)):
            print("WARNING! Not finite values in R Matrix")

    else:
        print("Already aligned!")
        r_ea = mean

    return r_ea


def compute_euclidean_alignment(data, mean=None, estimator="lwf", dtype="raw"):
    data = copy.deepcopy(data)

    r_ea = compute_ref_euclidean(data=data, mean=mean, estimator=estimator, dtype=dtype)

    if dtype == "raw":
        result = np.matmul(r_ea, data)
    else:
        result = r_ea @ data @ r_ea

    return result


def euclidean_alignment(X, size=24, domain=None, estimator="lwf", dtype="raw"):
    X_aux = []

    if domain is not None:
        for d in np.unique(domain):
            X_batch = X[domain == d]
            X_batch_EA = compute_euclidean_alignment(
                X_batch, estimator=estimator, dtype=dtype
            )
            X_aux.append(X_batch_EA)
        covmat_EA = np.concatenate(X_aux)
    else:
        if size is None:
            m = X.shape[0]
        else:
            m = size
        n = X.shape[0]

        for k in range(int(n / m)):
            X_batch = X[k * m : (k + 1) * m]
            X_batch_EA = compute_euclidean_alignment(
                X_batch, estimator=estimator, dtype=dtype
            )
            X_aux.append(X_batch_EA)
        covmat_EA = np.concatenate(X_aux)
    return covmat_EA
