import numpy as np
from numpy import any, iscomplexobj, isfinite, real
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import inv, sqrtm

from .covariance import compute_covariances


def compute_ref_oneclass(data_ref, dtype="covmat", estimator="lwf"):
    if dtype != "covmat":
        data_ref = compute_covariances(data_ref, estimator=estimator)

    mean = mean_covariance(data_ref, metric="riemann")

    compare = np.allclose(mean, np.identity(mean.shape[0]))

    if not compare:

        if iscomplexobj(mean):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(mean)):
            print("covariance matrix problem sqrt")

        ref = inv(sqrtm(mean))

        if iscomplexobj(ref):
            print(
                "WARNING! Covariance matrix was not SPD somehow. "
                + "Can be caused by running ICA-EOG rejection, if "
                + "not, check data!!"
            )
            ref = real(ref).astype(np.float64)
        elif not any(isfinite(ref)):
            print("WARNING! Not finite values in R Matrix")

    else:
        print("Already aligned!")
        ref = mean

    return ref


def compute_oneclass(data, y=None, ref=None, dtype="covmat", reflabel=0, estimator="lwf"):
    # Select the reference class

    if ref is None:
        class_ref = data[y == reflabel]
        ref = compute_ref_oneclass(class_ref, dtype=dtype, estimator=estimator)

    if dtype == "covmat":
        results = np.matmul(ref, data)
        align = np.matmul(results, ref)
    else:
        align = np.matmul(ref, data)

    return align


def one_class_alignment(data, y, size=None, domains=None, dtype="covmat", reflabel=0):
    if domains is None:  # Then needs to use group size info
        data_aux = []

        if size is None:
            m = data.shape[0]
        else:
            m = size
        n = data.shape[0]

        for k in range(int(n / m)):
            # For each domain
            data_group = data[k * m : (k + 1) * m]
            y_group = y[k * m : (k + 1) * m]
            # Align this group
            data_align = compute_oneclass(
                data_group, y_group, reflabel=reflabel, dtype=dtype
            )
            data_aux.append(data_align)
        aligned_data = np.concatenate(data_aux)

    else:
        aligned_data = np.zeros_like(data)
        for d in np.unique(domains):
            data_group = data[domains == d]
            y_group = y[domains == d]
            data_align = compute_oneclass(
                data_group, y_group, reflabel=reflabel, dtype=dtype
            )
            aligned_data[domains == d] = data_align

    return aligned_data
