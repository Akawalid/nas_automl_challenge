"""
A series of script to compare different methosds to compute the
new parameters with the low rank linear regression problem.
In particular compares Cholesky, LDL and SVD methods.
"""

from warnings import warn

import torch


def switch_preferred_linalg_library(
    func: callable,
) -> callable:
    """
    Decorator to switch the preferred linalg library for a function.
    """

    def new_func(*args, preferred_linalg_library: None | str = "cusolver", **kwargs):
        if preferred_linalg_library is not None:
            torch.backends.cuda.preferred_linalg_library(preferred_linalg_library)
        try:
            return func(*args, **kwargs)
        except torch.linalg.LinAlgError as e:
            if preferred_linalg_library == "cusolver":
                raise ValueError(
                    "This is probably a bug from CUDA < 12.1"
                    "Try torch.backends.cuda.preferred_linalg_library('magma')"
                )
            else:
                raise e

    return new_func


@switch_preferred_linalg_library
def sqrt_inverse_matrix_semi_positive(
    matrix: torch.Tensor,
    threshold: float = 1e-5,
) -> torch.Tensor:
    """
    Compute the square root of the inverse of a semi-positive definite matrix.

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix, square and semi-positive definite
    threshold: float
        threshold to consider an eigenvalue as zero
    preferred_linalg_library: None | str in ("magma", "cusolver")
        linalg library to use, "cusolver" may failed
        for non positive definite matrix if CUDA < 12.1 is used
        see: https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html

    Returns
    -------
    torch.Tensor
        square root of the inverse of the input matrix
    """
    assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."
    assert torch.allclose(matrix, matrix.t()), "The input matrix must be symmetric."
    assert torch.isnan(matrix).sum() == 0, "The input matrix must not contain NaN values."

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    selected_eigenvalues = eigenvalues > threshold
    eigenvalues = torch.rsqrt(eigenvalues[selected_eigenvalues])  # inverse square root
    eigenvectors = eigenvectors[:, selected_eigenvalues]
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()


def low_rank_factorization(
    matrix_s: torch.Tensor,
    matrix_n: torch.Tensor,
    pre_compute_inv_sqrt_operator: callable = sqrt_inverse_matrix_semi_positive,
    apply_transpose_inv_sqrt_operator: callable = torch.matmul,
    apply_inv_sqrt_operator: callable = torch.matmul,
    statistical_threshold: float = 1e-3,
    maximum_rank: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the optimal added parameters for a given layer.

    Compute H such that H^T H = S.
    Then compute the SVD of (H^+)^T N = U S V^T.
    The optimal added parameters are alpha = sign(S) sqrt(|S|) H^+ U
    and omega = sqrt(|S|) V.

    Parameters
    ----------
    matrix_s: torch.Tensor in (s, s)
        square matrix S
    matrix_n: torch.Tensor in (s, t)
        matrix N
    statistical_threshold: float
        threshold to consider an eigenvalue as zero in the SVD of H N
    apply_transpose_inv_sqrt_operator
        function to compute precompute(S), N -> (H^+)^T N
    apply_inv_sqrt_operator
        function to compute precompute(S), N -> H^+ N
    maximum_rank: int | None
        maximum rank, if None all significant ranks are kept
    pre_compute_inv_sqrt_operator: callable[torch.Tensor, **kwargs] -> torch.Tensor
        function to compute the square root of the inverse of S or
        anything such that apply_inv_sqrt_operator(pre_compute_inv_sqrt_operator(S), M) = H^+ M

    kwargs: dict
        additional arguments for the pre_compute_inv_sqrt_operator

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] in (k, s) (t, k) (k,)
        optimal added weights alpha, omega and eigenvalues lambda
    """
    # check the input matrices
    s_1, s_2 = matrix_s.shape
    assert s_1 == s_2, "The input matrix S must be square."
    n_1, n_2 = matrix_n.shape
    assert s_2 == n_1, (
        f"The input matrices S and N must have compatible shapes."
        f"(got {matrix_s.shape=} and {matrix_n.shape=})"
    )
    if not torch.allclose(matrix_s, matrix_s.t()):
        diff = torch.abs(matrix_s - matrix_s.t())
        warn(
            f"Warning: The input matrix S is not symmetric.\n"
            f"Max difference: {diff.max():.2e},"
            f"% of non-zero elements: {100 * (diff > 1e-10).sum() / diff.numel():.2f}%"
        )
        matrix_s = (matrix_s + matrix_s.t()) / 2

    # compute the square root of the inverse of S
    matrix_h = pre_compute_inv_sqrt_operator(matrix_s, **kwargs)
    # compute the product P := S^{-1/2} N
    matrix_p = apply_transpose_inv_sqrt_operator(matrix_h, matrix_n)

    # compute the SVD of the product
    try:
        u, s, v = torch.linalg.svd(matrix_p, full_matrices=False)
    except torch.linalg.LinAlgError:
        u, s, v = torch.linalg.svd(matrix_p, full_matrices=False)

    # select the singular values
    selected_singular_values = s >= min(statistical_threshold, s.max())
    if maximum_rank is not None:
        selected_singular_values[maximum_rank:] = False

    # keep only the significant singular values but keep at least one
    s = s[selected_singular_values]
    u = u[:, selected_singular_values]
    v = v[selected_singular_values, :]
    # compute the optimal added weights
    sqrt_s = torch.sqrt(torch.abs(s))
    alpha = torch.sign(s) * sqrt_s * (apply_inv_sqrt_operator(matrix_h, u))
    omega = sqrt_s[:, None] * v
    return alpha.t(), omega.t(), s


def compute_optimal_added_parameters(
    matrix_s: torch.Tensor,
    matrix_n: torch.Tensor,
    numerical_threshold: float = 1e-15,
    statistical_threshold: float = 1e-3,
    maximum_added_neurons: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the optimal added parameters for a given layer.

    Parameters
    ----------
    matrix_s: torch.Tensor in (s, s)
        square matrix S
    matrix_n: torch.Tensor in (s, t)
        matrix N
    numerical_threshold: float
        threshold to consider an eigenvalue as zero in the square root of the inverse of S
    statistical_threshold: float
        threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
    maximum_added_neurons: int | None
        maximum number of added neurons, if None all significant neurons are kept

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] in (k, s) (t, k) (k,)
        optimal added weights alpha, omega and eigenvalues lambda
    """
    return low_rank_factorization(
        matrix_s,
        matrix_n,
        pre_compute_inv_sqrt_operator=switch_preferred_linalg_library(
            sqrt_inverse_matrix_semi_positive
        ),
        apply_transpose_inv_sqrt_operator=torch.matmul,
        apply_inv_sqrt_operator=torch.matmul,
        statistical_threshold=statistical_threshold,
        maximum_rank=maximum_added_neurons,
        threshold=numerical_threshold,
    )


def low_rank_cholesky_invertible(
    matrix_s: torch.Tensor,
    matrix_n: torch.Tensor,
    statistical_threshold: float = 1e-3,
    maximum_added_neurons: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute new parameters using Cholesky decomposition.
    """
    return low_rank_factorization(
        matrix_s,
        matrix_n,
        pre_compute_inv_sqrt_operator=torch.linalg.cholesky,
        apply_transpose_inv_sqrt_operator=lambda h, m: torch.linalg.solve_triangular(
            h.t(), m, upper=True
        ),
        apply_inv_sqrt_operator=lambda h, m: torch.linalg.solve_triangular(
            h, m, upper=False
        ),
        statistical_threshold=statistical_threshold,
        maximum_rank=maximum_added_neurons,
    )


def positive_ldl(
    matrix_s: torch.Tensor, ratio_error_threshold: float = 2, zero_threshold: float = 1e-7
) -> torch.Tensor:
    """
    Compute the pseudo inverse square root of a positive semi-definite matrix using
     LDL decomposition.

    Parameters
    ----------
    matrix_s: torch.Tensor
        input matrix, square and positive semi-definite
    ratio_error_threshold: float
        maximum ratio between the error reconstruction with the threshold LDL decomposition
        and the error reconstruction with the original LDL
    zero_threshold: float
        minimum threshold to consider a diagonal element as zero

    Returns
    -------
    tuple[torch.Tensor]
        pseudo-inverse of H = sqrt(hat(D)) L^T with hat(D) >= 0
    """
    ld, _ = torch.linalg.ldl_factor(matrix_s, hermitian=False)
    diag = torch.diagonal(ld, dim1=-2, dim2=-1)
    lower = torch.tril(ld, diagonal=-1) + torch.eye(ld.shape[-1], device=ld.device)

    reconstruction_error_full = (
        torch.abs(matrix_s - lower @ diag @ lower.t()).max().item()
    )

    # zeros the diagonal elements below the threshold
    threshold = max(min(diag.min().item(), 0), zero_threshold)
    diag[torch.abs(diag) <= threshold] = 0

    reconstruction_error_threshold = (
        torch.abs(matrix_s - lower @ diag @ lower.t()).max().item()
    )

    if reconstruction_error_threshold / reconstruction_error_full > ratio_error_threshold:
        raise ValueError(
            "The reconstruction error with the threshold LDL decomposition is too high."
        )
    else:
        return torch.linalg.pinv(torch.diag(torch.sqrt(diag)) @ lower.t())


def low_rank_ldl(
    matrix_s: torch.Tensor,
    matrix_n: torch.Tensor,
    statistical_threshold: float = 1e-3,
    maximum_added_neurons: int | None = None,
    ratio_error_threshold: float = 2,
    zero_threshold: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute new parameters using LDL decomposition.
    """
    # here instead of inverting we could use the linalg.lstsq function
    return low_rank_factorization(
        matrix_s,
        matrix_n,
        pre_compute_inv_sqrt_operator=positive_ldl,
        apply_transpose_inv_sqrt_operator=lambda h, m: h.t() @ m,
        apply_inv_sqrt_operator=torch.matmul,
        statistical_threshold=statistical_threshold,
        maximum_rank=maximum_added_neurons,
        ratio_error_threshold=ratio_error_threshold,
        zero_threshold=zero_threshold,
    )
