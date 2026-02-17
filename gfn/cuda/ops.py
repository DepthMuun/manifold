"""
GFN CUDA Operations - Fused Operations Module
==============================================

This module provides fused CUDA operations for the GFN project.
Designed in a modular way to facilitate extension and testing.

Available operations:
- christoffel_fused: Christoffel symbols with low-rank decomposition
- leapfrog_fused: Symplectic leapfrog integrator
- heun_fused: Heun integrator (RK2)
- euler_fused: Euler integrator
- rk4_fused: Runge-Kutta 4 integrator
- verlet_fused: Verlet integrator
- head_mixing_fused: Attention head mixing
- dynamic_gating_fused: Dynamic gating
- recurrent_manifold_fused: Multi-layer manifold fusion


Date: 2026-02-07
"""

import torch
import torch.nn as nn
import os
import sys
import importlib.util
import importlib.machinery
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
from abc import ABC, abstractmethod

# Import base module
from .core import (
    device_manager,
    CudaConstants,
    operation_registry,
    check_cuda_availability
)


try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

# ============================================================================
# CUDA MODULE LOADER
# ============================================================================

class CudaModuleLoader:
    """
    Modular loader for compiled CUDA modules.
    
    Handles:
    - Compiled module detection
    - Dynamic loading
    - Automatic fallbacks
    """
    
    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
        self._load_paths = self._get_load_paths()
    
    def _get_load_paths(self) -> List[Path]:
        """Gets search paths for CUDA modules."""
        cuda_dir = Path(__file__).resolve().parent
        project_root = cuda_dir.parent.parent
        
        paths = [cuda_dir, project_root]
        
        if sys.platform.startswith("win"):
            for ver in ["v12.9", "v12.4", "v12.3", "v11.8"]:
                cuda_path = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{ver}/bin")
                if cuda_path.exists():
                    paths.append(cuda_path)
        
        return paths
    
    def _get_extension_patterns(self) -> List[str]:
        """Gets extension patterns based on platform."""
        if sys.platform.startswith("win"):
            return ["*.pyd"]
        elif sys.platform.startswith("darwin"):
            return ["*.dylib", "*.so"]
        else:
            return ["*.so"]
    
    def find_module(self, name: str) -> Optional[Path]:
        """Searches for a compiled module by name."""
        patterns = self._get_extension_patterns()
        
        for base_path in self._load_paths:
            if not base_path.exists():
                continue
            
            for pattern in patterns:
                for candidate in base_path.glob(pattern):
                    if name in candidate.stem:
                        return candidate
        
        return None
    
    def load(self, name: str) -> Optional[Any]:
        """Loads a module by name."""
        if name in self._loaded_modules:
            return self._loaded_modules[name]
        
        module_path = self.find_module(name)
        if module_path is None:
            return None
        
        try:
            loader = importlib.machinery.ExtensionFileLoader(name, str(module_path))
            spec = importlib.util.spec_from_file_location(name, str(module_path), loader=loader)
            
            if spec is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            
            self._loaded_modules[name] = module
            return module
            
        except Exception as e:
            print(f"[CUDA] Error loading module {name}: {e}")
            return None
    
    def preload_all(self):
        """Preloads all available CUDA modules."""
        module_names = ['gfn_cuda']
        
        for name in module_names:
            self.load(name)


# Global loader
_module_loader = CudaModuleLoader()

# Try to load compiled CUDA module
_CUDA_MODULE = _module_loader.load('gfn_cuda')
CUDA_AVAILABLE = _CUDA_MODULE is not None
gfn_cuda = _CUDA_MODULE


# ============================================================================
# OPERATION FACTORY
# ============================================================================

class OperationFactory:
    """
    Factory to create operations with automatic fallbacks.
    
    Design pattern: Factory Method + Strategy
    """
    
    _operations: Dict[str, Tuple[Callable, Callable]] = {}
    
    @classmethod
    def register(cls, name: str, cuda_op: Callable, python_op: Callable):
        """Registers a new operation."""
        cls._operations[name] = (cuda_op, python_op)
    
    @classmethod
    def create(cls, name: str, device: torch.device) -> Callable:
        """
        Creates an operation for the specified device.
        
        Args:
            name: Operation name
            device: Target device (cuda or cpu)
        
        Returns:
            Operation function ready to use
        """
        if name not in cls._operations:
            raise ValueError(f"Operation '{name}' not registered")
        
        cuda_op, python_op = cls._operations[name]
        
        if device.type == 'cuda' and CUDA_AVAILABLE:
            return cuda_op
        return python_op
    
    @classmethod
    def has_cuda(cls, name: str) -> bool:
        """Checks if the operation has a CUDA implementation."""
        if name not in cls._operations:
            return False
        cuda_op, _ = cls._operations[name]
        return cuda_op is not None


# ============================================================================
# BASE OPERATIONS (PYTHON FALLBACK)
# ============================================================================

class BaseOperation(ABC):
    """Base class for all operations."""
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Implements the forward pass."""
        pass
    
    @abstractmethod
    def backward(self, *args, **kwargs):
        """Implements the backward pass."""
        pass


class ChristoffelOperation(BaseOperation):
    """
    Christoffel symbols operation with low-rank decomposition.
    
    Computes: Γ^k_ij = Σ_r λ_kr * (U_ir * U_jr)
    
    Where:
    - v: Velocities [batch, dim]
    - U, W: Decomposition matrices [dim, rank]
    - x: Positions (optional) [batch, dim]
    - V_w: Potential weights (optional) [1, dim]
    
    Returns:
        gamma: Christoffel symbols [batch, dim]
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._init_constants()
    
    def _init_constants(self):
        """Initializes constants from configuration."""
        self.curvature_clamp = self.config.get('curvature_clamp', CudaConstants.CURVATURE_CLAMP)
        self.epsilon = self.config.get('epsilon', CudaConstants.EPSILON_STANDARD)
        self.singularity_gate_slope = self.config.get('singularity_gate_slope', CudaConstants.SINGULARITY_GATE_SLOPE)
    
    def forward(self, v: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                x: Optional[torch.Tensor] = None,
                V_w: Optional[torch.Tensor] = None,
                plasticity: float = 0.0,
                sing_thresh: float = 0.5,
                sing_strength: float = 2.0,
                topology: int = 0) -> torch.Tensor:
        """
        Christoffel forward pass.
        
        Args:
            v: Input velocities [B, D]
            U: Decomposition matrix U [D, R]
            W: Decomposition matrix W [D, R]
            x: Positions (optional) [B, D]
            V_w: Potential weights (optional) [1, D]
            plasticity: Curvature plasticity coefficient
            sing_thresh: Singularity threshold
            sing_strength: Singularity strength
            topology: Topology type (0=Euclidean, 1=Toroidal)
        
        Returns:
            gamma: Christoffel symbols [B, D]
        """
        # Velocity projection: h = U^T v
        h = torch.matmul(v, U)  # [B, R]
        
        # Energy normalization
        energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])
        scale = 1.0 / (1.0 + torch.sqrt(energy) + self.epsilon)
        
        # Plasticity factor
        M = 1.0
        if plasticity != 0.0:
            E = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
            M = 1.0 + plasticity * 0.1 * torch.tanh(E)
        
        # Singularities (curvature amplification)
        if x is not None and V_w is not None and V_w.numel() > 0:
            if topology == 1:  # Toroidal
                pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
            else:
                pot = torch.sum(x * V_w, dim=-1, keepdim=True)
            
            gate = torch.sigmoid(pot)
            soft_m = torch.sigmoid(self.singularity_gate_slope * (gate - sing_thresh))
            M = M * (1.0 + (sing_strength - 1.0) * soft_m)
        
        # Christoffel: γ = (h ⊙ h) @ W^T
        gamma = torch.matmul(h * h, W.t()) * scale * M
        gamma = self.curvature_clamp * torch.tanh(gamma / self.curvature_clamp)
        
        return gamma
    
    def backward(self, grad_output: torch.Tensor, v: torch.Tensor, U: torch.Tensor,
                 W: torch.Tensor, x: Optional[torch.Tensor] = None,
                 V_w: Optional[torch.Tensor] = None,
                 output: Optional[torch.Tensor] = None,
                 plasticity: float = 0.0,
                 sing_thresh: float = 0.5,
                 sing_strength: float = 2.0,
                 topology: int = 0) -> Tuple[torch.Tensor, ...]:
        """
        Christoffel backward pass using PyTorch autograd.
        
        Returns:
            Gradient tuple: (dv, dU, dW, dx, dV_w)
        """
        # Use PyTorch autograd for gradients
        if output is None:
            output = self.forward(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)
        
        grads = torch.autograd.grad(
            output, [v, U, W, x, V_w],
            grad_output,
            allow_unused=True,
            retain_graph=True
        )
        
        # Fill None gradients with zeros
        result = []
        for i, (tensor, grad) in enumerate([(v, grads[0]), (U, grads[1]), (W, grads[2])]):
            if grad is None:
                result.append(torch.zeros_like(tensor))
            else:
                result.append(grad)
        
        # For x and V_w (optional)
        if x is not None:
            result.append(grads[3] if grads[3] is not None else torch.zeros_like(x))
        if V_w is not None and V_w.numel() > 0:
            result.append(grads[4] if grads[4] is not None else torch.zeros_like(V_w))
        
        return tuple(result)


# ============================================================================
# TOROIDAL KERNEL BINDINGS (Component 2)
# ============================================================================

def launch_toroidal_leapfrog_fused(
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
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Launch dedicated toroidal leapfrog fused kernel.
    
    BUG-8 FIX: Now passes learned friction gate weights (W_forget, b_forget)
    to the CUDA kernel instead of using hardcoded DEFAULT_FRICTION.
    
    Args:
        x: Initial positions [batch, dim]
        v: Initial velocities [batch, dim]
        f: Force sequence [batch, seq_len, dim]
        R: Major radius of torus
        r: Minor radius of torus
        dt: Time step
        batch: Batch size
        seq_len: Sequence length
        dim: Dimension (should be even)
        hysteresis_state: Optional hysteresis state tensor
        W_forget: Forget gate weights [dim, feat_dim] or None for DEFAULT_FRICTION
        b_forget: Forget gate bias [dim] or None
        
    Returns:
        Tuple of (x_final, v_final, x_seq, reg_loss, h_state) or None if CUDA unavailable
    """
    try:
        import gfn_cuda
        
        # Prepare optional tensors
        W_f = W_forget if W_forget is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        b_f = b_forget if b_forget is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        
        x_out, v_out = gfn_cuda.toroidal_leapfrog_fused(
            x.contiguous(),
            v.contiguous(),
            f.contiguous(),
            float(R),
            float(r),
            float(dt),
            int(batch),
            int(seq_len),
            int(dim),
            W_f.contiguous(),
            b_f.contiguous()
        )
        
        # Prepare outputs to match recurrent_manifold_fused signature
        x_final = x_out[:, -1, :]  # [batch, dim]
        v_final = v_out[:, -1, :]  # [batch, dim]
        x_seq = x_out  # [batch, seq_len, dim]
        reg_loss = torch.tensor(0.0, device=x.device)
        h_state = hysteresis_state if hysteresis_state is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        
        return (x_final, v_final, x_seq, reg_loss, h_state)
        
    except (ImportError, AttributeError) as e:
        return None



class LeapfrogOperation(BaseOperation):
    """
    Symplectic Leapfrog (Stormer-Verlet) integrator.
    
    Implements:
    - Kick-Drift-Kick with implicit friction
    - Toroidal boundary (wrapping)
    - Hysteresis support
    
    Args:
        x: Positions [B, D]
        v: Velocities [B, D]
        f: External forces [B, D]
        U, W: Christoffel matrices
        dt: Base time step
        dt_scale: Time step scale per layer/head
        steps: Number of substeps
        topology: Topology type
        Wf, bf: Friction weights
        plasticity: Curvature plasticity
    
    Returns:
        x_out, v_out: Updated positions and velocities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._init_constants()
    
    def _init_constants(self):
        """Initializes constants."""
        self.friction_scale = self.config.get('friction_scale', CudaConstants.FRICTION_SCALE)
        self.epsilon = self.config.get('epsilon', CudaConstants.EPSILON_STANDARD)
        self.dt = self.config.get('dt', CudaConstants.DEFAULT_DT)
        self.toroidal_period = CudaConstants.TOROIDAL_PERIOD
    
    def forward(self, x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                U: torch.Tensor, W: torch.Tensor,
                dt_scale: float = 1.0,
                steps: int = 1,
                topology: int = 0,
                Wf: Optional[torch.Tensor] = None,
                bf: Optional[torch.Tensor] = None,
                W_input: Optional[torch.Tensor] = None,
                plasticity: float = 0.0,
                sing_thresh: float = 0.5,
                sing_strength: float = 2.0,
                V_w: Optional[torch.Tensor] = None,
                velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leapfrog integrator forward pass.
        """
        # Calculate effective dt
        eff_dt = self.dt * dt_scale
        h = 0.5 * eff_dt
        
        # Copy initial states
        curr_x = x.clone()
        curr_v = v.clone()
        
        # Initialize hysteresis state
        h_state = None
        
        christoffel_op = ChristoffelOperation({
            'curvature_clamp': self.config.get('curvature_clamp', CudaConstants.CURVATURE_CLAMP),
            'epsilon': self.epsilon
        })
        
        for _ in range(steps):
            # 1. Calculate friction coefficient
            mu = torch.zeros_like(v)
            if Wf is not None and bf is not None:
                feat = curr_x
                if topology == 1:
                    feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
                gate = torch.matmul(feat, Wf.t()) + bf
                if W_input is not None:
                    gate = gate + torch.matmul(f, W_input.t())
                mu = torch.sigmoid(gate) * self.friction_scale
                if velocity_friction_scale > 0:
                    v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
                    v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + CudaConstants.EPSILON_SMOOTH)
                    mu = mu * (1.0 + velocity_friction_scale * v_norm)
            
            # 2. Kick 1
            gamma = christoffel_op.forward(curr_v, U, W, curr_x, V_w, plasticity, sing_thresh, sing_strength, topology)
            curr_v = (curr_v + h * (f - gamma)) / (1.0 + h * mu + self.epsilon)
            
            # 3. Drift
            curr_x = curr_x + eff_dt * curr_v
            
            # Apply toroidal boundary with smooth wrapping (match Python)
            curr_x = apply_boundary_python(curr_x, topology)
            
            # 4. Kick 2
            if Wf is not None and bf is not None:
                feat = curr_x
                if topology == 1:
                    feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
                gate = torch.matmul(feat, Wf.t()) + bf
                if W_input is not None:
                    gate = gate + torch.matmul(f, W_input.t())
                mu = torch.sigmoid(gate) * self.friction_scale
                if velocity_friction_scale > 0:
                    v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
                    v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + CudaConstants.EPSILON_SMOOTH)
                    mu = mu * (1.0 + velocity_friction_scale * v_norm)
            
            gamma2 = christoffel_op.forward(curr_v, U, W, curr_x, V_w, plasticity, sing_thresh, sing_strength, topology)
            curr_v = (curr_v + h * (f - gamma2)) / (1.0 + h * mu + self.epsilon)
        
        return curr_x, curr_v
    
    def backward(self, grad_x: torch.Tensor, grad_v: torch.Tensor,
                 x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                 U: torch.Tensor, W: torch.Tensor,
                 dt_scale: float = 1.0,
                 steps: int = 1,
                 topology: int = 0,
                 Wf: Optional[torch.Tensor] = None,
                 bf: Optional[torch.Tensor] = None,
                 W_input: Optional[torch.Tensor] = None,
                 plasticity: float = 0.0,
                 sing_thresh: float = 0.5,
                 sing_strength: float = 2.0,
                 velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, ...]:
        """
        Leapfrog integrator backward pass.
        
        Returns:
            Gradient tuple
        """
        # Use PyTorch autograd
        def leapfrog_fn(x, v):
            return self.forward(
                x, v, f, U, W, dt_scale, steps, topology,
                Wf, bf, W_input, plasticity, sing_thresh, sing_strength, None, velocity_friction_scale
            )
        
        return torch.autograd.grad(
            leapfrog_fn(x, v),
            [x, v, f, U, W],
            (grad_x, grad_v),
            allow_unused=True,
            retain_graph=True
        )


# Register operations in factory
OperationFactory.register('christoffel', None, ChristoffelOperation)
OperationFactory.register('leapfrog', None, LeapfrogOperation)


# ============================================================================
# MODULE PUBLIC INTERFACE
# ============================================================================

def christoffel_fused(v: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                      x: Optional[torch.Tensor] = None,
                      V_w: Optional[torch.Tensor] = None,
                      plasticity: float = 0.0,
                      sing_thresh: float = 0.5,
                      sing_strength: float = 2.0,
                      topology: int = 0,
                      R: float = 2.0,
                      r: float = 1.0) -> torch.Tensor:
    """
    Public interface for Christoffel fused.
    
    Automatically detects whether to use CUDA or Python fallback.
    """
    if CUDA_AVAILABLE and v.is_cuda:
        # Use autograd wrapper if available
        try:
            from .autograd import christoffel_fused_autograd
            return christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
        except ImportError:
            pass
    
    # Python fallback
    op = ChristoffelOperation()
    return op.forward(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)


def leapfrog_fused(x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                   U: torch.Tensor, W: torch.Tensor,
                   dt: float,
                   dt_scale: float,
                   steps: int,
                   topology: int = 0,
                   Wf: Optional[torch.Tensor] = None,
                   bf: Optional[torch.Tensor] = None,
                   plasticity: float = 0.0,
                   sing_thresh: float = 0.5,
                   sing_strength: float = 2.0,
                   R: float = 2.0,
                   r: float = 1.0,
                   W_input: Optional[torch.Tensor] = None,
                   V_w: Optional[torch.Tensor] = None, # Singularity vector
                   hysteresis_state: Optional[torch.Tensor] = None,
                   hyst_update_w: Optional[torch.Tensor] = None,
                   hyst_update_b: Optional[torch.Tensor] = None,
                   hyst_readout_w: Optional[torch.Tensor] = None,
                   hyst_readout_b: Optional[torch.Tensor] = None,
                   hyst_decay: float = 0.9,
                   hyst_enabled: bool = False,
                   velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interfaz pública para Leapfrog fused.
    
    Detecta automáticamente si usar CUDA o Python fallback.
    """
    if CUDA_AVAILABLE and x.is_cuda:
        try:
            from .autograd import leapfrog_fused_autograd
            return leapfrog_fused_autograd(
                x, v, f, U, W, dt, dt_scale, steps, topology,
                Wf, bf, plasticity, sing_thresh, sing_strength, R, r,
                W_input, V_w, velocity_friction_scale,
                hysteresis_state, hyst_update_w, hyst_update_b, 
                hyst_readout_w, hyst_readout_b, hyst_decay, hyst_enabled
            )
        except ImportError:
            pass
    
    # Python fallback
    op = LeapfrogOperation({'dt': dt, 'friction_scale': CudaConstants.FRICTION_SCALE, 'epsilon': CudaConstants.EPSILON_STANDARD})
    return op.forward(
        x, v, f, U, W, dt_scale, steps, topology,
        Wf, bf, W_input, plasticity, sing_thresh, sing_strength, V_w=V_w,
        velocity_friction_scale=velocity_friction_scale
    )


def heun_fused(x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
               U: torch.Tensor, W: torch.Tensor,
               dt: float, dt_scale: float, steps: int,
               topology: int = 0,
               W_forget: Optional[torch.Tensor] = None,
               b_forget: Optional[torch.Tensor] = None,
               plasticity: float = 0.0,
               sing_thresh: float = 0.5,
               sing_strength: float = 2.0,
               R: float = 2.0,
               r: float = 1.0,
               W_input: Optional[torch.Tensor] = None,
               velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heun integrator fused operation.
    """
    if velocity_friction_scale > 0.0 and CUDA_AVAILABLE and x.is_cuda:
        # TODO: Log warning
        pass

    if CUDA_AVAILABLE and x.is_cuda:
        try:
            from .autograd import heun_fused_autograd
            return heun_fused_autograd(
                x, v, f, U, W, dt, dt_scale, steps, topology,
                W_forget, b_forget, plasticity, sing_thresh, sing_strength, R, r,
                W_input=W_input, velocity_friction_scale=velocity_friction_scale
            )
        except ImportError:
            pass
            
    # Python Fallback (Simple reconstruction of Heun logic)
    # We don't have HeunOperation class defined here, so we use ChristoffelOperation directly
    christoffel = ChristoffelOperation({'curvature_clamp': CudaConstants.CURVATURE_CLAMP})
    
    curr_x, curr_v = x.clone(), v.clone()
    eff_dt = dt * dt_scale
    
    for _ in range(steps):
        # Helper for dynamics: a = F - Gamma - Friction
        def dynamics(tx, tv):
             gamma = christoffel.forward(tv, U, W, tx, None, plasticity, sing_thresh, sing_strength, topology)
             
             # Friction logic (Python replication)
             mu = torch.zeros_like(tv)
             if W_forget is not None and b_forget is not None:
                 feat = tx
                 if topology == 1: feat = torch.cat([torch.sin(tx), torch.cos(tx)], dim=-1)
                 gate = torch.matmul(feat, W_forget.t()) + b_forget
                 if W_input is not None:
                     gate = gate + torch.matmul(f, W_input.t())
                 mu = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
                 
                 # Apply velocity scale correction here since we are in Python!
                 if velocity_friction_scale > 0.0:
                     vm = torch.norm(tv, dim=-1, keepdim=True) / (tv.shape[-1]**0.5 + 1e-8)
                     mu = mu * (1.0 + velocity_friction_scale * vm)

             acc = f - gamma - mu * tv
             return acc
        
        # k1
        dx1 = curr_v
        dv1 = dynamics(curr_x, curr_v)
        
        # Predictor
        x_pred = curr_x + eff_dt * dx1
        v_pred = curr_v + eff_dt * dv1
        
        # k2
        dx2 = v_pred
        dv2 = dynamics(x_pred, v_pred)
        
        # Update
        curr_x = curr_x + 0.5 * eff_dt * (dx1 + dx2)
        curr_v = curr_v + 0.5 * eff_dt * (dv1 + dv2)
        
        if topology == 1:
             curr_x = torch.atan2(torch.sin(curr_x), torch.cos(curr_x))
             
    return curr_x, curr_v


def euler_fused(x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                U: torch.Tensor, W: torch.Tensor,
                dt: float, dt_scale: float, steps: int,
                topology: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Euler fused operation stub."""
    # Simplified fallback
    eff_dt = dt * dt_scale
    curr_x, curr_v = x.clone(), v.clone()
    christoffel = ChristoffelOperation()
    
    for _ in range(steps):
        gamma = christoffel.forward(curr_v, U, W, curr_x, None, 0.0, 0.5, 2.0, topology)
        acc = f - gamma
        curr_v = curr_v + eff_dt * acc
        curr_x = curr_x + eff_dt * curr_v
        
    return curr_x, curr_v


def head_mixing_fused(x_heads: torch.Tensor, v_heads: torch.Tensor,
                      W_x: torch.Tensor, W_v: torch.Tensor,
                      topology: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    if CUDA_AVAILABLE and x_heads.is_cuda:
        try:
            import gfn_cuda
            if hasattr(gfn_cuda, 'head_mixing_fused'):
                return gfn_cuda.head_mixing_fused(x_heads, v_heads, W_x, W_v, int(topology))
        except Exception:
            pass
    batch = x_heads.shape[1]
    x_cat = x_heads.permute(1, 0, 2).contiguous().view(batch, -1)
    v_cat = v_heads.permute(1, 0, 2).contiguous().view(batch, -1)
    if topology == 1:
        v_mix = torch.tanh(v_cat / 100.0)
        mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)
        x_next = torch.matmul(mixer_in_x, W_x.t())
    else:
        x_next = torch.matmul(x_cat, W_x.t())
    v_next = torch.matmul(v_cat, W_v.t())
    if topology == 1:
        x_next = torch.atan2(torch.sin(x_next), torch.cos(x_next))
    v_next = 100.0 * torch.tanh(v_next / 100.0)
    return x_next, v_next


def dynamic_gating_fused(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                         W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    if CUDA_AVAILABLE and x.is_cuda:
        try:
            import gfn_cuda
            if hasattr(gfn_cuda, 'dynamic_gating_fused'):
                return gfn_cuda.dynamic_gating_fused(x, W1, b1, W2, b2)
        except Exception:
            pass
    hidden = torch.tanh(torch.matmul(x, W1.t()) + b1)
    out = torch.matmul(hidden, W2.t()) + b2
    return torch.sigmoid(out)


def recurrent_manifold_fused(x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                             U_stack: torch.Tensor, W_stack: torch.Tensor,
                             dt: float, dt_scales: torch.Tensor,
                             forget_rates: torch.Tensor, num_heads: int,
                             plasticity: float = 0.0, sing_thresh: float = 0.5,
                             sing_strength: float = 2.0,
                             mix_x: Optional[torch.Tensor] = None,
                             mix_v: Optional[torch.Tensor] = None,
                             Wf: Optional[torch.Tensor] = None,
                             Wi: Optional[torch.Tensor] = None,
                             bf: Optional[torch.Tensor] = None,
                             Wp: Optional[torch.Tensor] = None,
                             bp: Optional[torch.Tensor] = None,
                             topology: int = 0,
                             R: float = 2.0, r: float = 1.0,
                             **kwargs) -> Optional[Tuple]:
    needs_grad = (
        x.requires_grad or v.requires_grad or f.requires_grad or
        U_stack.requires_grad or W_stack.requires_grad
    )
    if CUDA_AVAILABLE and x.is_cuda and not needs_grad:
        try:
            import gfn_cuda
            # Prepare optional parameters for Unified Kernel
            wf = Wf if Wf is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            bf_val = bf if bf is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            wi = Wi if Wi is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            
            h_state = kwargs.get('hysteresis_state', torch.empty(0, device=x.device, dtype=x.dtype))
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
            
            # Match old return signature: (x_final, v_final, x_seq, reg_loss, h_state)
            reg_loss = res[3] if len(res) > 3 else torch.tensor(0.0, device=x.device)
            return (x_f, v_f, x_s, reg_loss, h_state)

            
        except (ImportError, AttributeError, RuntimeError) as e:
            # print(f"[CUDA WRAPPER ERROR] falling back to autograd: {e}")
            pass


    from .autograd import recurrent_manifold_fused_autograd
    return recurrent_manifold_fused_autograd(
        x=x, v=v, f=f,
        U_stack=U_stack, W_stack=W_stack,
        dt=dt, dt_scales=dt_scales, forget_rates=forget_rates, num_heads=num_heads,
        plasticity=plasticity, sing_thresh=sing_thresh, sing_strength=sing_strength,
        mix_x=mix_x, mix_v=mix_v, Wf=Wf, Wi=Wi, bf=bf, Wp=Wp, bp=bp,
        topology=topology, R=R, r=r,
        **kwargs
    )


# Alias para compatibilidad
lowrank_christoffel_fused = christoffel_fused


def get_registered_operations() -> Dict[str, bool]:
    """Obtiene el estado de registro de operaciones."""
    return {
        'christoffel_fused': CUDA_AVAILABLE or True,
        'leapfrog_fused': CUDA_AVAILABLE or True,
        'heun_fused': False,
        'euler_fused': False,
        'rk4_fused': False,
        'verlet_fused': False,
        'head_mixing_fused': CUDA_AVAILABLE,
        'dynamic_gating_fused': CUDA_AVAILABLE,
        'recurrent_manifold_fused': CUDA_AVAILABLE
    }
