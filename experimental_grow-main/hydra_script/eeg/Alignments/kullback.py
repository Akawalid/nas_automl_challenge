import copy

import numpy as np
from numpy import any, iscomplexobj, isfinite, real
from pyriemann.utils.mean import mean_kullback_sym
from scipy.linalg import inv, sqrtm

from .covariance import compute_covariances


# Compute Riemannian Mean of reference matrices
def compute_ref_kullback(data, dtype="covmat", estimator="lwf", sample_weight=None):

    if dtype != "covmat":
        data = compute_covariances(data, estimator=estimator)

    r_mix = mean_kullback_sym(data, sample_weight=sample_weight)

    compare = np.allclose(r_mix, np.identity(r_mix.shape[0]))

    if not compare:

        if iscomplexobj(r_mix):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(r_mix)):
            print("covariance matrix problem sqrt")

        r_kl = inv(sqrtm(r_mix))

        if iscomplexobj(r_kl):
            print(
                "WARNING! Covariance matrix was not SPD somehow. "
                + "Can be caused by running ICA-EOG rejection, if "
                + "not, check data!!"
            )
            r_kl = real(r_kl).astype(np.float64)
        elif not any(isfinite(r_kl)):
            print("WARNING! Not finite values in R Matrix")

    else:
        print("Already aligned!")
        r_kl = r_mix

    return r_kl


def compute_kullback(data, sample_weight=None, dtype="covmat", r_kl=None):
    # dtype (data type) could be covmat or raw
    data = copy.deepcopy(data)

    if r_kl is None:
        r_kl = compute_ref_kullback(data, dtype=dtype, sample_weight=sample_weight)

    if dtype == "covmat":
        # data is cov matrix
        result = np.matmul(r_kl, data)
        result = np.matmul(result, r_kl)

    else:
        # data is raw signal
        result = np.matmul(r_kl, data)

    return result


def kl_alignment(data, size=None, domains=None, sample_weight=None, dtype="covmat"):
    if domains is None:  # Then needs to use group size info
        data_aux = []

        if size is None:
            m = data.shape[0]
        else:
            m = size
        n = data.shape[0]

        for k in range(int(n / m)):
            data_group = data[k * m : (k + 1) * m]
            data_align = compute_kullback(
                data_group, sample_weight=sample_weight, dtype=dtype
            )
            data_aux.append(data_align)
        aligned_data = np.concatenate(data_aux)

    else:
        aligned_data = np.zeros_like(data)
        for d in np.unique(domains):
            print(d)
            data_group = data[domains == d]
            print(data_group.shape)
            data_align = compute_kullback(
                data_group, sample_weight=sample_weight, dtype=dtype
            )
            print(data_align.shape)
            aligned_data[domains == d] = data_align

    return aligned_data
