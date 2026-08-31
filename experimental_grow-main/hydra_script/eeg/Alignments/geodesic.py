import copy

import numpy as np
from numpy import any, iscomplexobj, isfinite, real
from pyriemann.utils.geodesic import geodesic_riemann
from scipy.linalg import inv, sqrtm


# Compute Riemannian Mean of reference matrices
def compute_ref_geodesic(m_ra, m_ea, data=None, alpha=0.5):

    r_mix = geodesic_riemann(m_ra, m_ea, alpha=alpha)

    compare = np.allclose(r_mix, np.identity(r_mix.shape[0]))

    if not compare:

        if iscomplexobj(r_mix):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(r_mix)):
            print("covariance matrix problem sqrt")

        r_hre = inv(sqrtm(r_mix))

        if iscomplexobj(r_hre):
            print(
                "WARNING! Covariance matrix was not SPD somehow. "
                + "Can be caused by running ICA-EOG rejection, if "
                + "not, check data!!"
            )
            r_hre = real(r_hre).astype(np.float64)
        elif not any(isfinite(r_hre)):
            print("WARNING! Not finite values in R Matrix")

    else:
        print("Already aligned!")
        r_hre = r_mix

    return r_hre


def compute_geodesic(data, m_ra=None, m_ea=None, alpha=0.5, dtype="covmat", r_hre=None):
    # dtype (data type) could be covmat or raw
    data = copy.deepcopy(data)

    if r_hre is None:
        r_hre = compute_ref_geodesic(m_ra, m_ea, data=data, alpha=alpha)

    if dtype == "covmat":
        # data is cov matrix
        result = np.matmul(r_hre, data)
        result = np.matmul(result, r_hre)

    else:
        # data is raw signal
        result = np.matmul(r_hre, data)

    return result


def geodesic_alignment(
    data, list_mean_r, list_mean_e, alpha=0.5, dtype="covmat", domains=None
):
    data_aux = []

    if domains is None:
        n_trials = data.shape[0]
        n_means = len(list_mean_r)
        m = int(n_trials / n_means)

        for k in range(int(n_means)):
            batch = data[k * m : (k + 1) * m]
            m_r = list_mean_r[k]
            m_e = list_mean_e[k]
            batch_HA = compute_geodesic(
                batch, m_ra=m_r, m_ea=m_e, alpha=alpha, dtype=dtype
            )
            data_aux.append(batch_HA)

        data_HA = np.concatenate(data_aux)

    else:
        for k in range(len(list_mean_r)):
            d = np.unique(domains)[k]
            batch = data[domains == d]
            m_r = list_mean_r[k]
            m_e = list_mean_e[k]
            batch_HA = compute_geodesic(
                batch, m_ra=m_r, m_ea=m_e, alpha=alpha, dtype=dtype
            )
            data_aux.append(batch_HA)

        data_HA = np.concatenate(data_aux)

    return data_HA
