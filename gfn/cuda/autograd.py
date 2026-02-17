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
            'curvature_clamp': CudaConstants.CURVATURE_CLAMP,
            'epsilon': CudaConstants.EPSILON_STANDARD,
            'singularity_gate_slope': CudaConstants.SINGULARITY_GATE_SLOPE
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
        
        if gamma is None:
            from .ops import ChristoffelOperation
            op = ChristoffelOperation({
                'curvature_clamp': CudaConstants.CURVATURE_CLAMP,
                'epsilon': CudaConstants.EPSILON_STANDARD,
                'singularity_gate_slope': CudaConstants.SINGULARITY_GATE_SLOPE
            })
            gamma = op.forward(v, U, W, x, V_w, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, ctx.topology)
        
        grad_inputs = torch.autograd.grad(
            gamma, [v, U, W, x, V_w], grad_output,
            allow_unused=True,
            retain_graph=False
        )
        
        result = []
        for i, (tensor, grad) in enumerate([
            (v, grad_inputs[0]),
            (U, grad_inputs[1]),
            (W, grad_inputs[2]),
            (x, grad_inputs[3]),
            (V_w, grad_inputs[4])
        ]):
            if grad is None:
                if tensor is not None and tensor.numel() > 0:
                    result.append(torch.zeros_like(tensor))
                else:
                    result.append(None)
            else:
                result.append(grad)
        
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
                W_input: Optional[torch.Tensor],
                V_w: Optional[torch.Tensor],
                velocity_friction_scale: float,
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
        if force is None:
            force = torch.zeros_like(x)
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
        if W_input is not None and W_input.numel() > 0:
            W_input = W_input.contiguous()
        else:
            W_input = torch.empty(0, device=x.device, dtype=x.dtype)
            
        if V_w is not None and V_w.numel() > 0:
            V_w = V_w.contiguous()
        else:
            V_w = torch.empty(0, device=x.device, dtype=x.dtype)
        
        # Try using CUDA
        if is_cuda:
            try:
                import gfn_cuda
                
                x_out, v_out = gfn_cuda.leapfrog_fused(
                    x, v, force, U, W,
                    float(dt), float(dt_scale), int(steps), int(topology),
                    Wf, bf, W_input, V_w,
                    float(plasticity), float(sing_thresh), float(sing_strength), float(R), float(r),
                    float(velocity_friction_scale),
                    hysteresis_state, hyst_update_w, hyst_update_b,
                    hyst_readout_w, hyst_readout_b, float(hyst_decay), hyst_enabled
                )
                
                ctx.save_for_backward(x, v, force, U, W, Wf, bf, x_out, v_out, V_w)
                ctx.dt = dt
                ctx.dt_scale = dt_scale
                ctx.steps = steps
                ctx.topology = topology
                ctx.plasticity = plasticity
                ctx.sing_thresh = sing_thresh
                ctx.sing_strength = sing_strength
                ctx.R = R
                ctx.r = r
                ctx.W_input = W_input
                ctx.velocity_friction_scale = velocity_friction_scale
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
        Returns gradients for all 26 forward arguments.
        """
        saved = ctx.saved_tensors
        x, v, force, U, W, Wf, bf, x_out, v_out, V_w = saved
        
        # 0:ctx, 1:x, 2:v, 3:f, 4:U, 5:W, 6:dt, 7:dt_s, 8:steps, 9:topo, 
        # 10:Wf, 11:bf, 12:plas, 13:th, 14:st, 15:R, 16:r, 17:Wi, 18:Vw, 19:vf
        # 20:h_s, 21:h_up_w, 22:h_up_b, 23:h_rd_w, 24:h_rd_b, 25:h_dec, 26:h_en
        
        if getattr(ctx, 'is_cuda', False):
            try:
                import gfn_cuda
                
                grads = gfn_cuda.leapfrog_backward_fused(
                    grad_x_out.contiguous(), grad_v_out.contiguous(),
                    x, v, force, U, W, Wf, bf, ctx.W_input, V_w,
                    float(ctx.dt), float(ctx.dt_scale), int(ctx.steps), int(ctx.topology),
                    float(ctx.plasticity), float(ctx.sing_thresh), float(ctx.sing_strength), float(ctx.R), float(ctx.r),
                    float(ctx.velocity_friction_scale),
                    ctx.hysteresis_state, ctx.hyst_update_w, ctx.hyst_update_b,
                    ctx.hyst_readout_w, ctx.hyst_readout_b, float(ctx.hyst_decay), ctx.hyst_enabled
                )
                
                # Unpack and map (grads returns 13 elements from CUDA)
                return (grads[0], grads[1], grads[2], grads[3], grads[4], # 1-5
                        None, None, None, None,                          # 6-9
                        grads[5], grads[6],                              # 10-11 (Wf, bf)
                        None, None, None, None, None,                    # 12-16 (scalars)
                        grads[7], grads[8],                              # 17-18 (Wi, Vw)
                        None,                                            # 19 (vf)
                        None,                                            # 20 (h_s)
                        grads[9], grads[10], grads[11], grads[12],       # 21-24 (h_params)
                        None, None)                                      # 25-26
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback: use PyTorch autograd
        def leapfrog_fn(x_in, v_in, force_in, U_in, W_in, Wf_in, bf_in, W_input_in, V_w_in):
            from .ops import LeapfrogOperation
            op = LeapfrogOperation({
                'dt': ctx.dt,
                'friction_scale': CudaConstants.FRICTION_SCALE,
                'epsilon': CudaConstants.EPSILON_STANDARD,
                'curvature_clamp': CudaConstants.CURVATURE_CLAMP
            })
            return op.forward(
                x_in, v_in, force_in, U_in, W_in, ctx.dt_scale, ctx.steps,
                ctx.topology,
                Wf_in if Wf_in.numel() > 0 else None,
                bf_in if bf_in.numel() > 0 else None,
                W_input_in if W_input_in.numel() > 0 else None,
                ctx.plasticity, ctx.sing_thresh, ctx.sing_strength,
                V_w_in if V_w_in.numel() > 0 else None,
                ctx.velocity_friction_scale
            )
        
        grads = torch.autograd.grad(
            leapfrog_fn(x, v, force, U, W, Wf, bf, ctx.W_input, V_w),
            [x, v, force, U, W, Wf, bf, ctx.W_input, V_w],
            (grad_x_out, grad_v_out),
            allow_unused=True,
            retain_graph=False
        )
        
        # Return exactly 26 values for PyTorch Fallback
        return (grads[0] if grads[0] is not None else torch.zeros_like(x),
                grads[1] if grads[1] is not None else torch.zeros_like(v),
                grads[2] if grads[2] is not None else torch.zeros_like(force),
                grads[3] if grads[3] is not None else torch.zeros_like(U),
                grads[4] if grads[4] is not None else torch.zeros_like(W),
                None, None, None, None,                          # 6-9
                grads[5] if grads[5] is not None else torch.zeros_like(Wf) if Wf.numel() > 0 else None,
                grads[6] if grads[6] is not None else torch.zeros_like(bf) if bf.numel() > 0 else None,
                None, None, None, None, None,                    # 12-16
                grads[7] if grads[7] is not None else torch.zeros_like(ctx.W_input) if ctx.W_input.numel() > 0 else None,
                grads[8] if grads[8] is not None else torch.zeros_like(V_w) if V_w.numel() > 0 else None,
                None,                                            # 19
                None, None, None, None, None,                    # 20-24
                None, None)                                      # 25-26


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
                            W_input: Optional[torch.Tensor] = None,
                            V_w: Optional[torch.Tensor] = None,
                            velocity_friction_scale: float = 0.0,
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
    """
    if Wf is None or not torch.is_tensor(Wf):
        Wf = torch.empty(0, device=x.device, dtype=x.dtype)
    if bf is None or not torch.is_tensor(bf):
        bf = torch.empty(0, device=x.device, dtype=x.dtype)
    if W_input is None or not torch.is_tensor(W_input):
        W_input = torch.empty(0, device=x.device, dtype=x.dtype)
    if V_w is None or not torch.is_tensor(V_w):
        V_w = torch.empty(0, device=x.device, dtype=x.dtype)
    
    # AUDIT FIX: Prepare hysteresis tensors
    if hysteresis_state is None or not torch.is_tensor(hysteresis_state):
        hysteresis_state = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_update_w is None or not torch.is_tensor(hyst_update_w):
        hyst_update_w = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_update_b is None or not torch.is_tensor(hyst_update_b):
        hyst_update_b = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_readout_w is None or not torch.is_tensor(hyst_readout_w):
        hyst_readout_w = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_readout_b is None or not torch.is_tensor(hyst_readout_b):
        hyst_readout_b = torch.empty(0, device=x.device, dtype=x.dtype)
    
    return LeapfrogAutogradFunction.apply(
        x, v, force, U, W, dt, dt_scale, steps, topology, Wf, bf, 
        plasticity, sing_thresh, sing_strength, R, r,
        W_input, V_w, velocity_friction_scale,
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
    Autograd fallback for recurrent manifold processing.
    Now supports full physics suite: Friction Gates, Torus, Hysteresis.
    """
    device = x.device
    B, D_total = x.shape
    T = f.shape[1]
    head_dim = D_total // num_heads
    num_layers = U_stack.shape[0] // num_heads # Should be 1 for a single MLayer
    
    x_curr = x.clone()
    v_curr = v.clone()
    x_seq = []
    
    # Hysteresis state initialization
    h_state = kwargs.get('hysteresis_state')
    if h_state is None and kwargs.get('hyst_enabled', False):
        h_state = torch.zeros_like(x)
    
    # Try using CUDA
    if x.is_cuda:
        try:
            import gfn_cuda
            
            # Prepare optional tensors
            wf = Wf if Wf is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            bf_val = bf if bf is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            wi = Wi if Wi is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            
            h_up_w = kwargs.get('hyst_update_w', torch.empty(0, device=x.device, dtype=x.dtype))
            h_up_b = kwargs.get('hyst_update_b', torch.empty(0, device=x.device, dtype=x.dtype))
            h_rd_w = kwargs.get('hyst_readout_w', torch.empty(0, device=x.device, dtype=x.dtype))
            h_rd_b = kwargs.get('hyst_readout_b', torch.empty(0, device=x.device, dtype=x.dtype))
            h_decay = float(kwargs.get('hyst_decay', 0.9))
            h_enabled = bool(kwargs.get('hyst_enabled', False))
            v_fric_scale = float(kwargs.get('velocity_friction_scale', 0.0))
            
            # Singularity Vector (Phase 2)
            v_w = kwargs.get('V_w', torch.empty(0, device=x.device, dtype=x.dtype))

            # Geometry Fusion (New in Phase 2)
            thermo_a = float(kwargs.get('thermo_alpha', 0.0))
            thermo_t = float(kwargs.get('thermo_temp', 1.0))
            h_z = kwargs.get('holographic_z', torch.empty(0, device=x.device, dtype=x.dtype))
            h_gz = kwargs.get('holographic_grad_z', torch.empty(0, device=x.device, dtype=x.dtype))

            # Unified kernel provides 100% parity including Torus, Gating, Hysteresis and Fused Geometry
            x_f, v_f, x_s = gfn_cuda.unified_mlayer_fused(
                x.contiguous(), v.contiguous(), f.contiguous(),
                U_stack.contiguous(), W_stack.contiguous(),
                wf.contiguous(), bf_val.contiguous(), wi.contiguous(),
                v_w.contiguous(),
                float(dt), dt_scales.contiguous(), int(topology),
                float(plasticity), float(sing_thresh), float(sing_strength),
                float(R), float(r), v_fric_scale,
                thermo_a, thermo_t, h_z.contiguous(), h_gz.contiguous(),
                h_state.contiguous(), h_up_w.contiguous(), h_up_b.contiguous(),
                h_rd_w.contiguous(), h_rd_b.contiguous(),
                h_decay, h_enabled
            )
            
            return x_f, v_f, x_s, torch.zeros((), device=device), h_state
            
        except (ImportError, AttributeError):
            pass
    
    # Reshape weights for head-parallel processing
    # U: [H, D_h, R], W: [H, R, D_h]
    U_heads = U_stack.view(num_heads, head_dim, -1)
    W_heads = W_stack.view(num_heads, head_dim, -1).permute(0, 2, 1)
    
    # Friction gates
    if Wf is not None:
        Wf_heads = Wf.view(num_heads, head_dim, -1)
        bf_heads = bf.view(num_heads, head_dim) if bf is not None else None
        Wi_heads = Wi.view(num_heads, head_dim, head_dim) if Wi is not None else None
    
    # Hysteresis parameters
    hyst_enabled = kwargs.get('hyst_enabled', False)
    if hyst_enabled:
        h_up_w = kwargs.get('hyst_update_w')
        h_up_b = kwargs.get('hyst_update_b')
        h_rd_w = kwargs.get('hyst_readout_w')
        h_rd_b = kwargs.get('hyst_readout_b')
        h_decay = kwargs.get('hyst_decay', 0.9)
        
    for t in range(T):
        f_t = f[:, t] # [B, D_total]
        
        # Split into heads for logic
        x_h = x_curr.view(B, num_heads, head_dim)
        v_h = v_curr.view(B, num_heads, head_dim)
        f_h = f_t.view(B, num_heads, head_dim)
        
        if h_state is not None:
            m_h = h_state.view(B, num_heads, head_dim)
        
        # Effective DT per head
        dt_eff = (dt * dt_scales).view(1, num_heads, 1) # [1, H, 1]
        h = 0.5 * dt_eff
        
        # 1. Evaluate Current Dynamics
        # Friction mu
        mu_h = torch.zeros_like(v_h)
        if Wf is not None:
            feat_h = x_h
            if topology == 1:
                feat_h = torch.cat([torch.sin(x_h), torch.cos(x_h)], dim=-1)
            
            gate = torch.einsum('bhd,hdk->bhk', feat_h, Wf_heads.transpose(1, 2))
            if bf_heads is not None:
                gate = gate + bf_heads.view(1, num_heads, head_dim)
            if Wi_heads is not None:
                gate = gate + torch.einsum('bhd,hdk->bhk', f_h, Wi_heads.transpose(1, 2))
            mu_h = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
            
            if kwargs.get('velocity_friction_scale', 0.0) > 0.0:
                v_norm = torch.norm(v_h, dim=-1, keepdim=True) / (head_dim**0.5 + 1e-8)
                mu_h = mu_h * (1.0 + kwargs['velocity_friction_scale'] * v_norm)
        
        # Ghost Force
        fg_h = torch.zeros_like(f_h)
        if hyst_enabled and h_rd_w is not None:
            fg_h = torch.einsum('bhd,hdk->bhk', m_h, h_rd_w.view(num_heads, head_dim, head_dim).transpose(1, 2))
            if h_rd_b is not None:
                fg_h = fg_h + h_rd_b.view(1, num_heads, head_dim)
        
        # Christoffel Gamma
        # h_proj: [B, H, R]
        h_proj = torch.einsum('bhd,hdr->bhr', v_h, U_heads)
        energy = torch.sum(h_proj * h_proj, dim=-1, keepdim=True) / max(1, h_proj.shape[-1])
        s_norm = 1.0 / (1.0 + torch.sqrt(energy) + 1e-8)
        
        # gamma: [B, H, D_h]
        gamma_h = torch.einsum('bhr,hrd->bhd', h_proj * h_proj, W_heads) * s_norm
        
        # --- NEW: Phase 2 Geometry Fusion in Autograd Fallback ---
        # A. Thermodynamic Modulation
        thermo_alpha = float(kwargs.get('thermo_alpha', 0.0))
        if thermo_alpha > 0.0:
            thermo_temp = float(kwargs.get('thermo_temp', 1.0))
            T = max(thermo_temp, 1e-8)
            # Energy E = (force^2).mean()
            f_energy = (f_h ** 2).mean(dim=-1, keepdim=True) # [B, H, 1]
            modulator = torch.exp(-thermo_alpha * f_energy / T)
            gamma_h = gamma_h * modulator
            
        # B. AdS/CFT Holographic Term
        holo_z = kwargs.get('holographic_z')
        holo_gz = kwargs.get('holographic_grad_z')
        if holo_z is not None and holo_gz is not None and holo_z.numel() > 0:
            # Reshape for heads
            z_h = holo_z.view(B, num_heads, 1)
            gz_h = holo_gz.view(B, num_heads, head_dim)
            
            v_dot_gz = (v_h * gz_h).sum(dim=-1, keepdim=True)
            v_sq = (v_h * v_h).sum(dim=-1, keepdim=True)
            
            gamma_ads = -(1.0 / (z_h + 1e-8)) * (2.0 * v_dot_gz * v_h - v_sq * gz_h)
            gamma_h = gamma_h + gamma_ads
        # --------------------------------------------------------

        gamma_h = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_h / CudaConstants.CURVATURE_CLAMP)
        
        # 2. Kick-Drift-Kick (Leapfrog Sequence)
        # Stage 1: Half-kick
        v_half = (v_h + h * (f_h + fg_h - gamma_h)) / (1.0 + h * mu_h + 1e-8)

        
        # Stage 2: Drift
        x_new_h = x_h + dt_eff * v_half
        if topology == 1:
            x_new_h = torch.atan2(torch.sin(x_new_h), torch.cos(x_new_h))
        
        # Stage 3: Re-evaluate and Final Kick
        # (For simplicity here, we assume mu and gamma are reasonably constant over dt for the second kick)
        # In the CUDA kernel we re-evaluate; let's do it here too for parity.
        
        # Re-eval mu/gamma at x_new_h
        if Wf is not None:
            feat_h = x_new_h
            if topology == 1:
                feat_h = torch.cat([torch.sin(x_new_h), torch.cos(x_new_h)], dim=-1)
            gate = torch.einsum('bhd,hdk->bhk', feat_h, Wf_heads.transpose(1, 2))
            if bf_heads is not None: gate = gate + bf_heads.view(1, num_heads, head_dim)
            if Wi_heads is not None: gate = gate + torch.einsum('bhd,hdk->bhk', f_h, Wi_heads.transpose(1, 2))
            mu_h = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
        
        h_proj = torch.einsum('bhd,hdr->bhr', v_half, U_heads)
        energy = torch.sum(h_proj * h_proj, dim=-1, keepdim=True) / max(1, h_proj.shape[-1])
        s_norm = 1.0 / (1.0 + torch.sqrt(energy) + 1e-8)
        gamma_h = torch.einsum('bhr,hrd->bhd', h_proj * h_proj, W_heads) * s_norm
        
        # --- NEW: Phase 2 Geometry Fusion in Autograd Fallback (Stage 2) ---
        if thermo_alpha > 0.0:
            gamma_h = gamma_h * modulator # Use same modulator as it depends on force_t
            
        if holo_z is not None and holo_gz is not None and holo_z.numel() > 0:
            gamma_ads = -(1.0 / (z_h + 1e-8)) * (2.0 * v_dot_gz * v_half - v_sq * gz_h)
            gamma_h = gamma_h + gamma_ads
        # -----------------------------------------------------------------

        gamma_h = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_h / CudaConstants.CURVATURE_CLAMP)
        
        v_new_h = (v_half + h * (f_h + fg_h - gamma_h)) / (1.0 + h * mu_h + 1e-8)

        
        # 3. Update State
        x_curr = x_new_h.reshape(B, D_total)
        v_curr = v_new_h.reshape(B, D_total)
        
        # Hysteresis State Update
        if hyst_enabled:
            h_in = x_new_h
            if topology == 1:
                h_in = torch.cat([torch.sin(x_new_h), torch.cos(x_new_h)], dim=-1)
            h_in = torch.cat([h_in, v_new_h], dim=-1)
            
            h_gate = torch.einsum('bhd,hdk->bhk', h_in, h_up_w.view(num_heads, head_dim, -1).transpose(1, 2))
            if h_up_b is not None:
                h_gate = h_gate + h_up_b.view(1, num_heads, head_dim)
            
            h_state = h_state * h_decay + torch.tanh(h_gate.reshape(B, D_total))
            
        x_seq.append(x_curr)
        
    x_seq = torch.stack(x_seq, dim=1)
    return x_curr, v_curr, x_seq, torch.zeros((), device=device), h_state



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




class HeunAutogradFunction(torch.autograd.Function):
    """
    Autograd function for fused Heun (RK2) integrator.
    """
    
    @staticmethod
    def forward(ctx, x, v, force, U, W, dt, dt_scale, steps, topology,
                W_forget, b_forget,
                plasticity, sing_thresh, sing_strength, R, r,
                W_input, velocity_friction_scale,
                hysteresis_state, hyst_update_w, hyst_update_b, 
                hyst_readout_w, hyst_readout_b, hyst_decay, hyst_enabled):
        
        is_cuda = x.is_cuda
        
        # Ensure contiguity
        x = x.contiguous()
        v = v.contiguous()
        if force is None: force = torch.zeros_like(x)
        force = force.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        
        wf = W_forget if W_forget is not None and torch.is_tensor(W_forget) and W_forget.numel() > 0 else torch.empty(0, device=x.device, dtype=x.dtype)
        bf = b_forget if b_forget is not None and torch.is_tensor(b_forget) and b_forget.numel() > 0 else torch.empty(0, device=x.device, dtype=x.dtype)
        wi = W_input if W_input is not None and torch.is_tensor(W_input) and W_input.numel() > 0 else torch.empty(0, device=x.device, dtype=x.dtype)
        
        # Try using CUDA
        if is_cuda:
            try:
                import gfn_cuda
                
                h_s = hysteresis_state if hysteresis_state is not None and torch.is_tensor(hysteresis_state) else torch.empty(0, device=x.device, dtype=x.dtype)
                h_up_w = hyst_update_w if hyst_update_w is not None and torch.is_tensor(hyst_update_w) else torch.empty(0, device=x.device, dtype=x.dtype)
                h_up_b = hyst_update_b if hyst_update_b is not None and torch.is_tensor(hyst_update_b) else torch.empty(0, device=x.device, dtype=x.dtype)
                h_rd_w = hyst_readout_w if hyst_readout_w is not None and torch.is_tensor(hyst_readout_w) else torch.empty(0, device=x.device, dtype=x.dtype)
                h_rd_b = hyst_readout_b if hyst_readout_b is not None and torch.is_tensor(hyst_readout_b) else torch.empty(0, device=x.device, dtype=x.dtype)

                x_out, v_out = gfn_cuda.heun_fused(
                    x, v, force, U, W,
                    float(dt), float(dt_scale), int(steps), int(topology),
                    wf, bf, wi,
                    float(plasticity), float(sing_thresh), float(sing_strength),
                    float(R), float(r), float(velocity_friction_scale),
                    h_s, h_up_w, h_up_b, h_rd_w, h_rd_b, float(hyst_decay), hyst_enabled
                )
                
                ctx.save_for_backward(x, v, force, U, W)
                ctx.dt = dt
                ctx.dt_scale = dt_scale
                ctx.steps = steps
                ctx.topology = topology
                ctx.R = R
                ctx.r = r
                ctx.is_cuda = True
                
                return x_out, v_out
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback to Python
        from .ops import heun_fused as heun_py
        x_out, v_out = heun_py(x, v, force, U, W, dt, dt_scale, steps, topology,
                               W_forget, b_forget, plasticity, sing_thresh, 
                               sing_strength, R, r, W_input, velocity_friction_scale)
        
        ctx.save_for_backward(x, v, force, U, W)
        ctx.dt = dt
        ctx.dt_scale = dt_scale
        ctx.steps = steps
        ctx.topology = topology
        ctx.R = R
        ctx.r = r
        ctx.is_cuda = False
        
        return x_out, v_out

    @staticmethod
    def backward(ctx, grad_x_out, grad_v_out):
        """
        Heun integrator backward pass.
        Returns gradients for all 25 forward arguments.
        """
        x, v, force, U, W = ctx.saved_tensors
        
        # 1:x, 2:v, 3:f, 4:U, 5:W, 6:dt, 7:dt_s, 8:steps, 9:topo, 
        # 10:Wf, 11:bf, 12:plas, 13:th, 14:st, 15:R, 16:r, 17:Wi, 18:vf
        # 19:h_s, 20:h_up_w, 21:h_up_b, 22:h_rd_w, 23:h_rd_b, 24:h_dec, 25:h_en
        
        if getattr(ctx, 'is_cuda', False):
            try:
                import gfn_cuda
                grads = gfn_cuda.heun_backward_fused(
                    grad_x_out.contiguous(), grad_v_out.contiguous(),
                    x, v, force, U, W,
                    float(ctx.dt), float(ctx.dt_scale), int(ctx.steps), int(ctx.topology),
                    float(ctx.R), float(ctx.r)
                )
                
                # Heun CUDA backward currently returns 5 grads: dx, dv, df, dU, dW
                return (grads[0], grads[1], grads[2], grads[3], grads[4], 
                        None, None, None, None, # 6-9
                        None, None,             # 10-11 (Wf, bf - not supported in Heun kernel yet)
                        None, None, None, None, None, # 12-16 (scalars)
                        None, None,             # 17-18 (Wi, vf)
                        None, None, None, None, None, None, None) # 19-25 (Hysteresis)
                        
            except (ImportError, AttributeError):
                pass
                
        # PyTorch Fallback
        def heun_fn(x_in, v_in, f_in, U_in, W_in):
            from .ops import heun_fused as heun_py
            return heun_py(x_in, v_in, f_in, U_in, W_in, 
                           ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology,
                           None, None, # Wf, bf
                           0.0, 0.5, 2.0, # plasticity, sing_thresh, sing_strength
                           ctx.R, ctx.r,
                           None, 0.0) # W_input, vf
        
        grads = torch.autograd.grad(
            heun_fn(x, v, force, U, W),
            [x, v, force, U, W],
            (grad_x_out, grad_v_out),
            allow_unused=True,
            retain_graph=False
        )
        
        return (grads[0] if grads[0] is not None else torch.zeros_like(x),
                grads[1] if grads[1] is not None else torch.zeros_like(v),
                grads[2] if grads[2] is not None else torch.zeros_like(force),
                grads[3] if grads[3] is not None else torch.zeros_like(U),
                grads[4] if grads[4] is not None else torch.zeros_like(W),
                None, None, None, None, # 6-9
                None, None,             # 10-11
                None, None, None, None, None, # 12-16
                None, None,             # 17-18
                None, None, None, None, None, None, None) # 19-25


class LowRankChristoffelWithFrictionFunction(torch.autograd.Function):
    """
    Legacy autograd function for Christoffel with Friction.
    """
    @staticmethod
    def forward(ctx, v, U, W, x, V_w, force, W_forget, b_forget, W_input,
                plasticity, sing_thresh, sing_strength, topology, R, r,
                velocity_friction_scale=0.0):
        
        ctx.save_for_backward(v, U, W, x, V_w, force, W_forget, b_forget, W_input)
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        ctx.topology = topology
        ctx.R = R
        ctx.r = r
        ctx.velocity_friction_scale = velocity_friction_scale
        
        from .ops import christoffel_fused
        gamma = christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
        
        mu = torch.zeros_like(v)
        if W_forget is not None and torch.is_tensor(W_forget) and b_forget is not None and torch.is_tensor(b_forget):
            features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) if topology == 1 else x
            mu = torch.sigmoid(torch.matmul(features, W_forget.t()) + b_forget) * CudaConstants.FRICTION_SCALE
            if velocity_friction_scale > 0:
                mu = mu * (1.0 + velocity_friction_scale * (torch.norm(v, dim=-1, keepdim=True) / (v.shape[-1]**0.5 + 1e-8)))
        
        return gamma + mu * v

    @staticmethod
    def backward(ctx, grad_output):
        v, U, W, x, V_w, force, W_f, b_f, W_i = ctx.saved_tensors
        
        if v.is_cuda:
            try:
                import gfn_cuda
                grads = gfn_cuda.lowrank_christoffel_friction_backward(
                    grad_output.contiguous(), torch.empty(0), v, U, W, x, V_w, force,
                    W_f if W_f is not None and torch.is_tensor(W_f) else torch.empty(0),
                    b_f if b_f is not None and torch.is_tensor(b_f) else torch.empty(0),
                    W_i if W_i is not None and torch.is_tensor(W_i) else torch.empty(0),
                    float(ctx.plasticity), float(ctx.sing_thresh), float(ctx.sing_strength),
                    int(ctx.topology), float(ctx.R), float(ctx.r), float(ctx.velocity_friction_scale)
                )
                return (grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], grads[7], grads[8],
                        None, None, None, None, None, None, None)
            except (ImportError, AttributeError):
                pass
        return (None,) * 16


def heun_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology=0,
                       W_forget=None, b_forget=None, 
                       plasticity=0.0, sing_thresh=0.5, sing_strength=2.0,
                       R=2.0, r=1.0, 
                       W_input=None, velocity_friction_scale=0.0,
                       **kwargs):
    h_s = kwargs.get('hysteresis_state')
    h_up_w = kwargs.get('hyst_update_w')
    h_up_b = kwargs.get('hyst_update_b')
    h_rd_w = kwargs.get('hyst_readout_w')
    h_rd_b = kwargs.get('hyst_readout_b')
    h_decay = kwargs.get('hyst_decay', 0.9)
    h_enabled = kwargs.get('hyst_enabled', False)
    
    return HeunAutogradFunction.apply(
        x, v, force, U, W, dt, dt_scale, steps, topology,
        W_forget, b_forget,
        plasticity, sing_thresh, sing_strength, R, r,
        W_input, velocity_friction_scale,
        h_s, h_up_w, h_up_b, h_rd_w, h_rd_b, h_decay, h_enabled
    )


def toroidal_leapfrog_fused_autograd(
    x: torch.Tensor, 
    v: torch.Tensor, 
    f: torch.Tensor,
    R: float,
    r: float,
    dt: float,
    batch: int,
    seq_len: int,
    dim: int,
    hysteresis_state: Optional[torch.Tensor] = None,
    W_forget: Optional[torch.Tensor] = None,
    b_forget: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Toroidal leapfrog integration with autograd support.
    
    This function implements toroidal manifold integration in pure PyTorch
    to support autograd during training. It computes metric-derived Christoffel
    symbols for toroidal geometry and performs leapfrog integration.
    
    Args:
        x: Initial positions [batch, dim]
        v: Initial velocities [batch, dim]
        f: Force sequence [batch, seq_len, dim]
        R: Major radius of torus
        r: Minor radius of torus
        dt: Time step
        batch: Batch size
        seq_len: Sequence length
        dim: Dimension (must be even for toroidal pairs)
        hysteresis_state: Optional hysteresis state
        
    Returns:
        Tuple of (x_final, v_final, x_seq, reg_loss, h_state)
    """
    device = x.device
    dtype = x.dtype
    
    # Initialize output sequences
    x_seq = []
    x_curr = x.clone()
    v_curr = v.clone()
    
    # Process sequence
    for t in range(seq_len):
        force_t = f[:, t]  # [batch, dim]
        
        # Friction gate (first half-step)
        if W_forget is not None and b_forget is not None and W_forget.shape[0] == dim and W_forget.shape[1] == 2 * dim and b_forget.shape[0] == dim:
            feats = torch.cat([torch.sin(x_curr), torch.cos(x_curr)], dim=-1)
            gate = feats @ W_forget.t() + b_forget
            mu1 = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
        else:
            mu1 = torch.zeros_like(x_curr)
        
        v_half = (v_curr + 0.5 * dt * force_t) / (1.0 + 0.5 * dt * mu1 + CudaConstants.EPSILON_STANDARD)
        
        # Position update with toroidal Christoffel correction
        x_new = x_curr + dt * v_half
        
        # Apply toroidal boundary conditions using atan2
        # This ensures smooth gradients through the wrapping
        x_new = torch.atan2(torch.sin(x_new), torch.cos(x_new))
        
        # Compute toroidal Christoffel symbols at new position
        gamma = torch.zeros_like(x_new)
        for i in range(0, dim, 2):  # Process pairs (θ, φ)
            if i + 1 < dim:
                theta = x_new[:, i]
                phi = x_new[:, i + 1]
                v_theta = v_half[:, i]
                v_phi = v_half[:, i + 1]
                
                # Metric coefficient: g_φφ = (R + r*cos(θ))²
                cos_theta = torch.cos(theta)
                denom = torch.clamp(R + r * cos_theta, min=1e-6)
                
                # Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r
                sin_theta = torch.sin(theta)
                gamma_theta = denom * sin_theta / (r + 1e-6) * (v_phi * v_phi)
                
                # Γ^φ_θφ = -r*sin(θ) / (R + r*cos(θ))
                gamma_phi = -(r * sin_theta) / (denom + 1e-6) * 2.0 * v_theta * v_phi
                
                gamma[:, i] = gamma_theta
                gamma[:, i + 1] = gamma_phi
        
        gamma = torch.clamp(gamma, -10.0, 10.0)
        
        # Friction gate (second half-step)
        if W_forget is not None and b_forget is not None and W_forget.shape[0] == dim and W_forget.shape[1] == 2 * dim and b_forget.shape[0] == dim:
            feats2 = torch.cat([torch.sin(x_new), torch.cos(x_new)], dim=-1)
            gate2 = feats2 @ W_forget.t() + b_forget
            mu2 = torch.sigmoid(gate2) * CudaConstants.FRICTION_SCALE
        else:
            mu2 = torch.zeros_like(x_curr)
        
        v_new = (v_half + 0.5 * dt * (force_t - gamma)) / (1.0 + 0.5 * dt * mu2 + CudaConstants.EPSILON_STANDARD)
        
        # Update state
        x_curr = x_new
        v_curr = v_new
        
        x_seq.append(x_curr)
    
    # Stack sequence
    x_seq = torch.stack(x_seq, dim=1)  # [batch, seq_len, dim]
    
    # Return final state and sequence
    reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
    h_state = hysteresis_state if hysteresis_state is not None else torch.empty(0, device=device, dtype=dtype)
    
    return x_curr, v_curr, x_seq, reg_loss, h_state
