import torch
import math
from typing import Tuple

# Try to import CUDA kernels
try:
    from .cuda import cuda_ops
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def parallel_scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes y_t = a_t * y_{t-1} + x_t via parallel associative scan.
    
    Args:
        a: Multiplicative term [batch, seq_len, dim]
        x: Additive term [batch, seq_len, dim]
        
    Returns:
        y: Scan result [batch, seq_len, dim]
        
    Raises:
        ValueError: If input tensors have incorrect dimensions or shapes
    """
    # Input validation
    if a.dim() != 3 or x.dim() != 3:
        raise ValueError(
            f"Expected 3D tensors [batch, seq_len, dim], "
            f"got a.shape={a.shape}, x.shape={x.shape}"
        )
    
    if a.shape != x.shape:
        raise ValueError(
            f"Input tensors must have same shape, "
            f"got a.shape={a.shape}, x.shape={x.shape}"
        )
    
    B, L, D = x.shape
    
    if L <= 0:
        raise ValueError(f"Sequence length must be positive, got L={L}")
    
    # Use CUDA kernel if available and on GPU
    if CUDA_AVAILABLE and a.is_cuda:
        try:
            return cuda_ops.parallel_scan_fused(a, x)
        except Exception as e:
            print(f"[GFN:WARN] CUDA parallel_scan failed: {e}, falling back to PyTorch")
            # Fall through to PyTorch implementation
    
    # Fallback to PyTorch implementation
    # For MANIFOLD, we are solving v_t = decay_t * v_{t-1} + force_t
    # Algorithm: y_t = x_t + a_t * y_{t-1}
    # This is a first-order linear recurrence.
    
    if L < 32:
        # Sequential is faster for tiny/short sequences
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(L):
            h = a[:, t] * h + x[:, t]
            y[:, t] = h
        return y

    # Recursive Doubling (Hillis-Steele) - O(log N) depth, O(N log N) work
    # This is NOT work-efficient but is parallel depth efficient.
    # Good for modern GPUs with massive parallelism.
    
    curr_a = a.clone()
    curr_x = x.clone()
    
    # Calculate number of steps: log2(L)
    steps = int(math.ceil(math.log2(L)))
    
    for i in range(steps):
        shift = 2**i
        
        # Shifted values
        prev_a = torch.roll(curr_a, shifts=shift, dims=1)
        prev_x = torch.roll(curr_x, shifts=shift, dims=1)
        
        # Mask out wrapped around elements
        # We only want to combine with elements strictly 'before' us in time.
        # torch.roll wraps around, so indices [0, ... shift-1] get values from end.
        # We must mask them.
        
        mask = torch.ones(L, device=x.device, dtype=x.dtype)
        mask[:shift] = 0
        mask = mask.view(1, L, 1)
        
        # Matrix multiplication in log-space/linear space of the update operator
        # New operator composition:
        # (a2, x2) o (a1, x1) = (a2*a1, a2*x1 + x2)
        
        # Update
        new_a = curr_a * prev_a
        new_x = curr_a * prev_x + curr_x
        
        # Apply mask: for t < shift, we don't change
        curr_a = torch.where(mask > 0.5, new_a, curr_a)
        curr_x = torch.where(mask > 0.5, new_x, curr_x)
        
    return curr_x
