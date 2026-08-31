import copy

import numpy as np
from numpy import any, iscomplexobj, isfinite, real
from pyriemann.utils.geodesic import geodesic_riemann
from scipy.linalg import inv, sqrtm

from .covariance import compute_covariances, mean_covariance


def compute_ref_riemann(data=None, mean=None, dtype="covmat"):
    if dtype != "covmat":
        covmats = compute_covariances(data, estimator="lwf")
        data = covmats

    if mean is None:
        mean = mean_covariance(data, metric="riemann")

    compare = np.allclose(mean, np.identity(mean.shape[0]))

    if not compare:

        if iscomplexobj(mean):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(mean)):
            print("covariance matrix problem sqrt")

        r_ra = inv(sqrtm(mean))

        if iscomplexobj(r_ra):
            print(
                "WARNING! Covariance matrix was not SPD somehow. "
                + "Can be caused by running ICA-EOG rejection, if "
                + "not, check data!!"
            )
            r_ra = real(r_ra).astype(np.float64)
        elif not any(isfinite(r_ra)):
            print("WARNING! Not finite values in R Matrix")

    else:
        print("Already aligned!")
        r_ra = mean

    return r_ra


def compute_riemannian_alignment(data, mean=None, dtype="covmat"):
    data = copy.deepcopy(data)

    r_ra = compute_ref_riemann(data=data, mean=mean, dtype=dtype)

    if dtype == "covmat":
        result = np.matmul(r_ra, data)
        result = np.matmul(result, r_ra)
    else:
        result = np.matmul(r_ra, data)

    return result


def riemannian_alignment(covmats, size=24, domain=None, dtype="covmat"):
    covmats_aux = []
    if domain is not None:
        for d in np.unique(domain):
            print(covmats.shape)
            print(domain.shape)
            batch = covmats[domain == d]
            batch_RA = compute_riemannian_alignment(batch, dtype=dtype)
            covmats_aux.append(batch_RA)
        covmats_RA = np.concatenate(covmats_aux)
    else:
        if size is None:
            m = covmats.shape[0]
        else:
            m = size
        n = covmats.shape[0]
        for k in range(int(n / m)):
            batch = covmats[k * m : (k + 1) * m]
            batch_RA = compute_riemannian_alignment(batch, dtype=dtype)
            covmats_aux.append(batch_RA)
        covmats_RA = np.concatenate(covmats_aux)
    return covmats_RA


def compute_resting_alignment(data, covmats_rest, r_ra=None, mean=None, dtype="covmat"):
    data = copy.deepcopy(data)

    if r_ra is None:
        r_ra = compute_ref_riemann(data=covmats_rest, mean=mean)

    if dtype == "raw":
        result = np.matmul(r_ra, data)

    else:
        result = np.matmul(r_ra, data)
        result = np.matmul(result, r_ra)

    return result


def riemannian_resting_alignment(covmats, covmats_rest, size=24, dtype="covmat"):
    covmats_aux = []
    if size is None:
        m = covmats.shape[0]
    else:
        m = size
    n = covmats.shape[0]
    for k in range(int(n / m)):
        batch = covmats[k * m : (k + 1) * m]
        rest = covmats_rest[k * m : (k + 1) * m]
        batch_RA = compute_resting_alignment(batch, rest, dtype=dtype)
        covmats_aux.append(batch_RA)
    covmats_RA = np.concatenate(covmats_aux)
    return covmats_RA


def compute_hyb_resting_alignment(
    data, covmats_rest, alpha=0.5, r_ra=None, mean=None, dtype="covmat"
):
    data = copy.deepcopy(data)

    if r_ra is None:
        m_ra = mean_covariance(covmats_rest, metric="riemann")
        m_ea = mean_covariance(covmats_rest, metric="euclid")
        r_mix = geodesic_riemann(m_ra, m_ea, alpha=alpha)
        r_ra = inv(sqrtm(r_mix))

    if dtype == "raw":
        result = np.matmul(r_ra, data)

    else:
        result = np.matmul(r_ra, data)
        result = np.matmul(result, r_ra)

    return result


def riemannian_hyb_resting_alignment(
    covmats, covmats_rest, alpha=0.5, size=24, dtype="covmat"
):
    covmats_aux = []
    if size is None:
        m = covmats.shape[0]
    else:
        m = size
    n = covmats.shape[0]
    for k in range(int(n / m)):
        batch = covmats[k * m : (k + 1) * m]
        rest = covmats_rest[k * m : (k + 1) * m]
        batch_RA = compute_hyb_resting_alignment(batch, rest, alpha=alpha, dtype=dtype)
        covmats_aux.append(batch_RA)
    covmats_RA = np.concatenate(covmats_aux)
    return covmats_RA
