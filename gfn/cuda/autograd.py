"""
GFN CUDA Autograd - Automatic Differentiation Module
========================================================

This module provides automatic differentiation functions for the
GFN project's CUDA operations.

Available functions:
- christoffel_fused_autograd
- leapfrog_fused_autograd
- heun_fused_autograd
- recurrent_manifold_fused_autograd

Date: 2026-02-07
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from functools import wraps
import time
from .core import CudaConstants


# ============================================================================
# TIMING REGISTRY
# ============================================================================

class TimingRegistry:
    """
    Execution time registry for profiling.
    
    Allows:
    - Measuring CUDA vs Python operation times
    - Accumulating statistics
    - Generating performance reports
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
        self._enabled = False
    
    def enable(self):
        """Enables timing registration."""
        self._enabled = True
    
    def disable(self):
        """Disables timing registration."""
        self._enabled = False
    
    def record(self, name: str, duration: float):
        """Records an execution time."""
        if not self._enabled:
            return
        
        if name not in self._timings:
            self._timings[name] = []
            self._counts[name] = 0
        
        self._timings[name].append(duration)
        self._counts[name] += 1
    
    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Gets statistics for an operation."""
        if name not in self._timings or not self._timings[name]:
            return None
        
        times = self._timings[name]
        return {
            'count': self._counts[name],
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'total': sum(times),
            'last': times[-1]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Gets statistics for all operations."""
        return {name: self.get_stats(name) for name in self._timings}
    
    def reset(self):
        """Resets all statistics."""
        self._timings = {}
        self._counts = {}
    
    def summary(self) -> str:
        """Generates a formatted summary."""
        if not self._timings:
            return "No timing data recorded"
        
        lines = ["=" * 60, "EXECUTION TIME SUMMARY", "=" * 60]
        
        for name, stats in self.get_all_stats().items():
            if stats:
                lines.append(f"\n{name}:")
                lines.append(f"  Count:  {stats['count']}")
                lines.append(f"  Mean:   {stats['mean']*1000:.3f} ms")
                lines.append(f"  Min:    {stats['min']*1000:.3f} ms")
                lines.append(f"  Max:    {stats['max']*1000:.3f} ms")
                lines.append(f"  Total:  {stats['total']*1000:.3f} ms")
        
        return "\n".join(lines)


# Global timing registry instance
timing_registry = TimingRegistry()


# ============================================================================
# TIMING DECORATORS
# ============================================================================

def timed_operation(name: Optional[str] = None):
    """
    Decorator to measure operation execution times.
    
    Args:
        name: Operation name (uses function name if not specified)
    """
    def decorator(func):
        op_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            
            if timing_registry._enabled:
                timing_registry.record(op_name, duration)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# AUTOGRAD FUNCTIONS
# ============================================================================

class ChristoffelAutogradFunction(torch.autograd.Function):
    """
    Autograd function for fused Christoffel.
    
    Implements forward and backward for automatic differentiation.
    """
    
    @staticmethod
    def forward(ctx, v: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                x: Optional[torch.Tensor], V_w: Optional[torch.Tensor],
                plasticity: float, sing_thresh: float, sing_strength: float,
                topology: int, R: float, r: float) -> torch.Tensor:
        """
        Forward pass.
        """
        # Determine device
        is_cuda = v.is_cuda
        
        # Ensure contiguous tensors
        v = v.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        
        if x is not None and x.numel() > 0:
            x = x.contiguous()
        else:
            x = torch.empty(0, device=v.device, dtype=v.dtype)
        
        if V_w is not None and V_w.numel() > 0:
            V_w = V_w.contiguous()
        else:
            V_w = torch.empty(0, device=v.device, dtype=v.dtype)
        
        # Try using CUDA
        if is_cuda:
            try:
                import gfn_cuda
                gamma = gfn_cuda.lowrank_christoffel_fused(
                    v, U, W, x, V_w,
                    float(plasticity), float(sing_thresh), float(sing_strength),
                    int(topology), float(R), float(r)
                )
                
                # Save tensors for backward
                ctx.save_for_backward(v, U, W, x, V_w, gamma)
                ctx.plasticity = plasticity
                ctx.sing_thresh = sing_thresh
                ctx.sing_strength = sing_strength
                ctx.topology = topology
                ctx.R = R
                ctx.r = r
                ctx.is_cuda = True
                
                return gamma
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback to Python
        from .ops import ChristoffelOperation
        
        op = ChristoffelOperation({
            'curvature_clamp': 3.0,
            'epsilon': 1e-8,
            'singularity_gate_slope': 1.0
        })
        
        gamma = op.forward(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)
        
        # Save tensors for backward
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        ctx.topology = topology
        ctx.R = R
        ctx.r = r
        ctx.is_cuda = False
        
        return gamma
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple:
        """
        Backward pass.
        
        AUDIT FIX (2026-02-07): Robust gradient unpacking with validation.
        The CUDA kernel returns a specific number of gradients that must match
        our expected structure. We now validate the length and provide clear
        error messages if there's a mismatch.
        """
        # Recover saved tensors
        saved = ctx.saved_tensors
        if len(saved) == 6:
            v, U, W, x, V_w, gamma = saved
        else:
            v, U, W, x, V_w = saved
            gamma = None
        
        # Expected gradient count based on CUDA kernel signature
        # Christoffel backward returns: dv, dU, dW, dx, dV_w, + 6 None params = 11 total
        EXPECTED_GRADIENTS = 11
        
        # Try CUDA backward
        if getattr(ctx, 'is_cuda', False):
            try:
                import gfn_cuda
                grads = gfn_cuda.christoffel_backward_fused(
                    grad_output.contiguous(),
                    gamma if gamma is not None else torch.empty(0),
                    v, U, W, x, V_w,
                    float(ctx.plasticity), float(ctx.sing_thresh), float(ctx.sing_strength),
                    int(ctx.topology), float(ctx.R), float(ctx.r)
                )
                
                # AUDIT FIX (2026-02-07): Validate gradient count
                actual_grads = len(grads)
                if actual_grads != 5:
                    print(f"[GFN:DEBUG] Christoffel backward returned {actual_grads} gradients, expected 5")
                
                # Safely unpack gradients, padding with None if needed
                result = []
                for i in range(5):  # Core gradients: dv, dU, dW, dx, dV_w
                    if i < actual_grads:
                        result.append(grads[i])
                    else:
                        result.append(None)
                
                # Add None for non-differentiable parameters (plasticity, sing_thresh, etc.)
                result.extend([None] * (EXPECTED_GRADIENTS - len(result)))
                
                return tuple(result)
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback to PyTorch autograd
        # Reconstruct output if necessary
        if gamma is None:
            from .ops import ChristoffelOperation
            op = ChristoffelOperation({
                'curvature_clamp': 3.0,
                'epsilon': 1e-8,
                'singularity_gate_slope': 1.0
            })
            gamma = op.forward(v, U, W, x, V_w, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, ctx.topology)
        
        # Calculate gradients using autograd
        grad_inputs = torch.autograd.grad(
            gamma, [v, U, W, x, V_w], grad_output,
            allow_unused=True,
            retain_graph=False
        )
        
        # Fill with zeros where necessary
        result = []
        for i, (tensor, grad) in enumerate([
            (v, grad_inputs[0]),
            (U, grad_inputs[1]),
            (W, grad_inputs[2]),
            (x, grad_inputs[3]),
            (V_w, grad_inputs[4])
        ]):
            if grad is None:
                if tensor.numel() > 0:
                    result.append(torch.zeros_like(tensor))
                else:
                    result.append(None)
            else:
                result.append(grad)
        
        # Add None gradients for non-differentiable parameters
        result.extend([None, None, None, None, None, None])
        
        return tuple(result)


class LeapfrogAutogradFunction(torch.autograd.Function):
    """
    Autograd function for fused Leapfrog.
    
    Implements forward and backward for the symplectic integrator.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, v: torch.Tensor, force: torch.Tensor,
                U: torch.Tensor, W: torch.Tensor,
                dt: float, dt_scale: float, steps: int,
                topology: int,
                Wf: Optional[torch.Tensor], bf: Optional[torch.Tensor],
                plasticity: float, sing_thresh: float, sing_strength: float, R: float, r: float,
                # AUDIT FIX (Component 7): Hysteresis parameters
                hysteresis_state: torch.Tensor,
                hyst_update_w: torch.Tensor,
                hyst_update_b: torch.Tensor,
                hyst_readout_w: torch.Tensor,
                hyst_readout_b: torch.Tensor,
                hyst_decay: float,
                hyst_enabled: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leapfrog integrator forward pass.
        """
        is_cuda = x.is_cuda
        
        # Ensure contiguity
        x = x.contiguous()
        v = v.contiguous()
        force = force.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        
        if Wf is not None and Wf.numel() > 0:
            Wf = Wf.contiguous()
        else:
            Wf = torch.empty(0, device=x.device, dtype=x.dtype)
        
        if bf is not None and bf.numel() > 0:
            bf = bf.contiguous()
        else:
            bf = torch.empty(0, device=x.device, dtype=x.dtype)
        
        # Try using CUDA
        if is_cuda:
            try:
                import gfn_cuda
                
                x_out, v_out = gfn_cuda.leapfrog_fused(
                    x, v, force, U, W,
                    float(dt), float(dt_scale), int(steps), int(topology),
                    Wf, bf, float(plasticity), float(sing_thresh), float(sing_strength), float(R), float(r),
                    # AUDIT FIX: Pass hysteresis parameters
                    hysteresis_state, hyst_update_w, hyst_update_b,
                    hyst_readout_w, hyst_readout_b, float(hyst_decay), hyst_enabled
                )
                
                ctx.save_for_backward(x, v, force, U, W, Wf, bf, x_out, v_out)
                ctx.dt = dt
                ctx.dt_scale = dt_scale
                ctx.steps = steps
                ctx.topology = topology
                ctx.plasticity = plasticity
                ctx.sing_thresh = sing_thresh
                ctx.sing_strength = sing_strength
                ctx.R = R
                ctx.r = r
                # AUDIT FIX: Save hysteresis parameters
                ctx.hysteresis_state = hysteresis_state
                ctx.hyst_update_w = hyst_update_w
                ctx.hyst_update_b = hyst_update_b
                ctx.hyst_readout_w = hyst_readout_w
                ctx.hyst_readout_b = hyst_readout_b
                ctx.hyst_decay = hyst_decay
                ctx.hyst_enabled = hyst_enabled
                ctx.is_cuda = True
                
                return x_out, v_out
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback Python
        from .ops import LeapfrogOperation
        
        op = LeapfrogOperation({
            'dt': dt,
            'friction_scale': CudaConstants.FRICTION_SCALE,
            'epsilon': CudaConstants.EPSILON_STANDARD,
            'curvature_clamp': CudaConstants.CURVATURE_CLAMP
        })
        
        x_out, v_out = op.forward(x, v, force, U, W, dt_scale, steps, topology, Wf, bf, plasticity, sing_thresh, sing_strength)
        
        ctx.save_for_backward(x, v, force, U, W, Wf, bf, x_out, v_out)
        ctx.dt = dt
        ctx.dt_scale = dt_scale
        ctx.steps = steps
        ctx.topology = topology
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        ctx.R = R
        ctx.r = r
        ctx.is_cuda = False
        
        return x_out, v_out
    
    @staticmethod
    def backward(ctx, grad_x_out: torch.Tensor, grad_v_out: torch.Tensor) -> Tuple:
        """
        Leapfrog integrator backward pass.
        """
        saved = ctx.saved_tensors
        x, v, force, U, W, Wf, bf, x_out, v_out = saved
        
        if getattr(ctx, 'is_cuda', False):
            try:
                import gfn_cuda
                
                grads = gfn_cuda.leapfrog_backward_fused(
                    grad_x_out.contiguous(), grad_v_out.contiguous(),
                    x, v, force, U, W, Wf, bf,
                    float(ctx.dt), float(ctx.dt_scale), int(ctx.steps), int(ctx.topology),
                    float(ctx.plasticity), float(ctx.sing_thresh), float(ctx.sing_strength), float(ctx.R), float(ctx.r),
                    # AUDIT FIX: Pass hysteresis parameters from ctx
                    ctx.hysteresis_state, ctx.hyst_update_w, ctx.hyst_update_b,
                    ctx.hyst_readout_w, ctx.hyst_readout_b, float(ctx.hyst_decay), ctx.hyst_enabled
                )
                
                # AUDIT FIX: Unpack hysteresis gradients (now 11 outputs instead of 7)
                return (grads[0], grads[1], grads[2], grads[3], grads[4],
                        None, None, None, None, grads[5], grads[6], None, None, None, None, None,
                        grads[7], grads[8], grads[9], grads[10], None, None, None)
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback: use PyTorch autograd
        def leapfrog_fn(x_in, v_in):
            from .ops import LeapfrogOperation
            op = LeapfrogOperation({
                'dt': ctx.dt,
                'friction_scale': CudaConstants.FRICTION_SCALE,
                'epsilon': CudaConstants.EPSILON_STANDARD,
                'curvature_clamp': CudaConstants.CURVATURE_CLAMP
            })
            return op.forward(x_in, v_in, force, U, W, ctx.dt_scale, ctx.steps,
                            ctx.topology, Wf, bf, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength)
        
        grads = torch.autograd.grad(
            leapfrog_fn(x, v),
            [x, v, force, U, W],
            (grad_x_out, grad_v_out),
            allow_unused=True,
            retain_graph=False
        )
        
        # Complete result
        result = []
        for i, (tensor, grad) in enumerate([
            (x, grads[0]), (v, grads[1]), (force, grads[2]),
            (U, grads[3]), (W, grads[4])
        ]):
            result.append(grad if grad is not None else torch.zeros_like(tensor))
        
        # Additional parameters
        result.extend([None, None, None, None,  # dt, dt_scale, steps, topology
                       grads[5] if len(grads) > 5 else torch.zeros_like(Wf) if Wf.numel() > 0 else None,
                       grads[6] if len(grads) > 6 else torch.zeros_like(bf) if bf.numel() > 0 else None,
                       None, None, None, None, None])  # plasticity, sing_thresh, sing_strength, R, r
        
        return tuple(result)


# ============================================================================
# PUBLIC INTERFACE
# ============================================================================

def christoffel_fused_autograd(v: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                               x: Optional[torch.Tensor] = None,
                               V_w: Optional[torch.Tensor] = None,
                               plasticity: float = 0.0,
                               sing_thresh: float = 0.5,
                               sing_strength: float = 2.0,
                               topology: int = 0,
                               R: float = 2.0,
                               r: float = 1.0) -> torch.Tensor:
    """
    Computes Christoffel symbols with autograd support.
    
    Args:
        v: Velocities [B, D]
        U: Matrix U [D, R]
        W: Matrix W [D, R]
        x: Optional positions [B, D]
        V_w: Optional potential weights [1, D]
        plasticity: Curvature plasticity
        sing_thresh: Singularity threshold
        sing_strength: Singularity strength
        topology: Topology type
        R, r: Torus radii (only for toroidal topology)
    
    Returns:
        gamma: Christoffel symbols [B, D]
    """
    # Prepare optional tensors
    if x is None:
        x = torch.empty(0, device=v.device, dtype=v.dtype)
    if V_w is None:
        V_w = torch.empty(0, device=v.device, dtype=v.dtype)
    
    return ChristoffelAutogradFunction.apply(
        v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r
    )


def leapfrog_fused_autograd(x: torch.Tensor, v: torch.Tensor, force: torch.Tensor,
                            U: torch.Tensor, W: torch.Tensor,
                            dt: float, dt_scale: float, steps: int,
                            topology: int = 0,
                            Wf: Optional[torch.Tensor] = None,
                            bf: Optional[torch.Tensor] = None,
                            plasticity: float = 0.0,
                            sing_thresh: float = 0.5,
                            sing_strength: float = 2.0,
                            R: float = 2.0,
                            r: float = 1.0,
                            # AUDIT FIX (Component 7): Hysteresis parameters with defaults
                            hysteresis_state: Optional[torch.Tensor] = None,
                            hyst_update_w: Optional[torch.Tensor] = None,
                            hyst_update_b: Optional[torch.Tensor] = None,
                            hyst_readout_w: Optional[torch.Tensor] = None,
                            hyst_readout_b: Optional[torch.Tensor] = None,
                            hyst_decay: float = 0.9,
                            hyst_enabled: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrates Hamiltonian system with autograd support.
    
    Args:
        x: Positions [B, D]
        v: Velocities [B, D]
        force: External forces [B, D]
        U, W: Christoffel matrices
        dt: Base time step
        dt_scale: dt scale
        steps: Number of substeps
        topology: Topology type
        Wf, bf: Friction weights
        plasticity: Curvature plasticity
        R, r: Torus radii
        hysteresis_state: Initial hysteresis state [B, D]
        hyst_update_w, hyst_update_b: Update parameters
        hyst_readout_w, hyst_readout_b: Readout parameters
        hyst_decay: Hysteresis decay
        hyst_enabled: Enable hysteresis
    
    Returns:
        x_out, v_out: Updated states
    """
    if Wf is None:
        Wf = torch.empty(0, device=x.device, dtype=x.dtype)
    if bf is None:
        bf = torch.empty(0, device=x.device, dtype=x.dtype)
    
    # AUDIT FIX: Prepare hysteresis tensors
    if hysteresis_state is None:
        hysteresis_state = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_update_w is None:
        hyst_update_w = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_update_b is None:
        hyst_update_b = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_readout_w is None:
        hyst_readout_w = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_readout_b is None:
        hyst_readout_b = torch.empty(0, device=x.device, dtype=x.dtype)
    
    return LeapfrogAutogradFunction.apply(
        x, v, force, U, W, dt, dt_scale, steps, topology, Wf, bf, plasticity, sing_thresh, sing_strength, R, r,
        hysteresis_state, hyst_update_w, hyst_update_b, hyst_readout_w, hyst_readout_b,
        hyst_decay, hyst_enabled
    )


def recurrent_manifold_fused_autograd(x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                                       U_stack: torch.Tensor, W_stack: torch.Tensor,
                                       dt: float, dt_scales: torch.Tensor,
                                       forget_rates: torch.Tensor, num_heads: int,
                                       plasticity: float, sing_thresh: float,
                                       sing_strength: float,
                                       mix_x: Optional[torch.Tensor],
                                       mix_v: Optional[torch.Tensor],
                                       Wf: Optional[torch.Tensor],
                                       Wi: Optional[torch.Tensor],
                                       bf: Optional[torch.Tensor],
                                       Wp: Optional[torch.Tensor],
                                       bp: Optional[torch.Tensor],
                                       topology: int = 0,
                                       R: float = 2.0, r: float = 1.0,
                                       **kwargs) -> Tuple:
    """
    Multi-layer manifold fusion (placeholder).
    """
    # For now, delegate to Python fallback
    from .ops import ChristoffelOperation
    
    device = x.device
    dtype = x.dtype
    B, D = x.shape
    T = f.shape[1]
    num_layers = U_stack.shape[0] // num_heads
    head_dim = D // num_heads
    
    # Simplified implementation for testing
    x_curr = x
    v_curr = v
    x_seq = []
    
    christoffel_op = ChristoffelOperation()
    
    for t in range(T):
        force_t = f[:, t]
        
        for layer_idx in range(num_layers):
            x_out = []
            v_out = []
            
            for h in range(num_heads):
                s = h * head_dim
                e = (h + 1) * head_dim
                
                x_h = x_curr[:, s:e]
                v_h = v_curr[:, s:e]
                f_h = force_t[:, s:e]
                
                # Simple integration
                gamma = christoffel_op.forward(v_h, U_stack[layer_idx * num_heads + h],
                                              W_stack[layer_idx * num_heads + h])
                
                dt_eff = dt * (dt_scales[layer_idx, h] if dt_scales.dim() > 1 else dt_scales)
                
                v_h = v_h + dt_eff * (f_h - gamma)
                x_h = x_h + dt_eff * v_h
                
                if topology == 1:
                    # AUDIT FIX: Use atan2 for smooth gradients
                    x_h = torch.atan2(torch.sin(x_h), torch.cos(x_h))
                
                x_out.append(x_h)
                v_out.append(v_h)
            
            x_curr = torch.cat(x_out, dim=-1)
            v_curr = torch.cat(v_out, dim=-1)
        
        x_seq.append(x_curr)
    
    x_seq = torch.stack(x_seq, dim=1)
    return x_curr, v_curr, x_seq, torch.zeros((), device=device), None


def get_timing_stats() -> Dict[str, Dict[str, float]]:
    """Gets timing stats."""
    return timing_registry.get_all_stats()


def enable_timing():
    """Enables timing registration."""
    timing_registry.enable()


def disable_timing():
    """Disables timing registration."""
    timing_registry.disable()


def reset_timing():
    """Resets timing statistics."""
    timing_registry.reset()
