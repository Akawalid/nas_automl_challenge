from math import ceil

import cuda.tile as ct
import torch

# Type alias for compile-time constants
ConstInt = ct.Constant[int]


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current CUDA block (CTA) in a 1D grid.
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Get the global IDs of a given CUDA block in a 1D grid.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


# Step 1: Define the kernel
@ct.kernel
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    # 1.1 Get block ID and map to output tile position
    # inside swizzle_2d, we access ct.bid(0) and output bidx and bidy
    M = A.shape[0]
    N = B.shape[1]
    GROUP_SIZE_M = 8
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    # 1.2 Calculate the number of tiles along the K dimension
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # 1.3 Initialize accumulator
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

    # Optional: promote fp32 inputs to tf32 for Tensor Core acceleration
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # 1.4 Loop over K dimension
    for k in range(num_tiles_k):
        # Load tiles from A and B
        a = ct.load(
            A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO
        ).astype(dtype)
        b = ct.load(
            B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO
        ).astype(dtype)

        # Matrix multiply-accumulate
        accumulator = ct.mma(a, b, accumulator)

    # 1.5 Store result
    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


# Step 2: Launch the kernel
def cutile_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Determine tile sizes based on dtype
    if A.dtype.itemsize == 2:  # float16/bfloat16
        tm, tn, tk = 128, 256, 64
    else:  # float32
        tm, tn, tk = 32, 32, 32

    m, k = A.shape
    _, n = B.shape

    # Calculate grid dimensions
    grid_x = ceil(m / tm)
    grid_y = ceil(n / tn)
    grid_size = grid_x * grid_y
    grid = (grid_size, 1, 1)

    # Create output tensor
    C = torch.empty((m, n), device=A.device, dtype=A.dtype)

    # Launch kernel
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, C, tm, tn, tk))

    return C


A = torch.rand(1000, 256)
B = torch.rand(256, 800)
C = cutile_matmul(A, B)
print(C.shape)
