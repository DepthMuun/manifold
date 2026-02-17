"""
GFN CUDA Numerical Suite
========================
Consolidated suite for:
- Numerical accuracy (Python vs CUDA)
- Gradient verification (autograd.gradcheck)
- Physical property preservation

Usage: pytest tests/cuda/test_cuda_numerical_suite.py
"""

import torch
import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.cuda.ops import (
    christoffel_fused,
    leapfrog_fused,
    heun_fused,
    ChristoffelOperation,
    LeapfrogOperation,
    CUDA_AVAILABLE
)
from gfn.cuda.autograd import (
    christoffel_fused_autograd,
    leapfrog_fused_autograd,
    heun_fused_autograd,
    LowRankChristoffelWithFrictionFunction
)
from gfn.geometry.lowrank import LowRankChristoffel
from gfn.integrators.symplectic.leapfrog import LeapfrogIntegrator
from gfn.integrators.runge_kutta.heun import HeunIntegrator
from gfn.constants import FRICTION_SCALE, DEFAULT_DT, LEAPFROG_SUBSTEPS

# Shared constants for suite
BATCH_SIZE = 8
DIM = 32
RANK = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDANumericalAccuracy:
    """Rigorous Python vs CUDA accuracy tests."""

    @pytest.fixture
    def test_data(self):
        torch.manual_seed(42)
        v = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
        x = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
        U = torch.randn(DIM, RANK, device=DEVICE, dtype=torch.float64)
        W = torch.randn(DIM, RANK, device=DEVICE, dtype=torch.float64)
        force = torch.randn(BATCH_SIZE, DIM, device=DEVICE, dtype=torch.float64)
        return v, x, U, W, force

    def test_christoffel_accuracy(self, test_data):
        v, x, U, W, _ = test_data
        
        # Python
        physics_config = {
            'topology': {'type': 'euclidean'},
            'stability': {'curvature_clamp': 20.0, 'friction': 0.0}
        }
        christoffel_py = LowRankChristoffel(dim=DIM, rank=RANK, physics_config=physics_config).to(DEVICE)
        christoffel_py.U.data = U
        christoffel_py.W.data = W
        
        with torch.no_grad():
            gamma_py = christoffel_py(v, None)
            
        # CUDA
        gamma_cuda = christoffel_fused(v, U, W, torch.empty(0, device=DEVICE, dtype=torch.float64), 
                                      torch.empty(0, device=DEVICE, dtype=torch.float64),
                                      0.0, 1.0, 1.0, 0, 2.0, 1.0)
        
        torch.testing.assert_close(gamma_py, gamma_cuda, rtol=1e-12, atol=1e-13)

    def test_leapfrog_accuracy(self, test_data):
        v_init, x_init, U, W, force = test_data
        dt = 0.05
        steps = 5
        
        # Python
        physics_config = {'topology': {'type': 'euclidean'}, 'stability': {'curvature_clamp': 20.0}}
        christoffel_py = LowRankChristoffel(dim=DIM, rank=RANK, physics_config=physics_config).to(DEVICE)
        christoffel_py.U.data = U
        christoffel_py.W.data = W
        integrator_py = LeapfrogIntegrator(christoffel=christoffel_py, dt=dt)
        
        with torch.no_grad():
            x_py, v_py = integrator_py.forward(x_init.clone(), v_init.clone(), force, steps=steps)
            
        # CUDA
        x_cuda, v_cuda = leapfrog_fused(x_init.clone(), v_init.clone(), force, U, W, dt, 1.0, steps, 0,
                                       torch.empty(0, device=DEVICE, dtype=torch.float64),
                                       torch.empty(0, device=DEVICE, dtype=torch.float64),
                                       0.0, 2.0, 1.0)
        
        torch.testing.assert_close(x_py, x_cuda, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(v_py, v_cuda, rtol=1e-12, atol=1e-13)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAGradients:
    """Verification of analytical gradients using gradcheck."""

    def test_christoffel_gradcheck(self):
        dtype = torch.float64
        v = torch.randn(4, 16, device=DEVICE, dtype=dtype, requires_grad=True)
        U = torch.randn(16, 4, device=DEVICE, dtype=dtype, requires_grad=True)
        W = torch.randn(16, 4, device=DEVICE, dtype=dtype, requires_grad=True)
        x = torch.randn(4, 16, device=DEVICE, dtype=dtype, requires_grad=True)
        V_w_fixed = torch.randn(16, device=DEVICE, dtype=dtype)
        
        def func(v_i, U_i, W_i, x_i):
            return christoffel_fused(v_i, U_i, W_i, x_i, V_w_fixed,
                                    0.5, 0.5, 2.0, 1, 2.0, 1.0)
                                    
        torch.autograd.gradcheck(func, (v, U, W, x), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_leapfrog_gradcheck(self):
        dtype = torch.float64
        B, D, R = 4, 8, 4
        x = torch.randn(B, D, device=DEVICE, dtype=dtype, requires_grad=True)
        v = torch.randn(B, D, device=DEVICE, dtype=dtype, requires_grad=True)
        f = torch.randn(B, D, device=DEVICE, dtype=dtype, requires_grad=True)
        U = torch.randn(D, R, device=DEVICE, dtype=dtype, requires_grad=True)
        W = torch.randn(D, R, device=DEVICE, dtype=dtype, requires_grad=True)
        Wf = torch.randn(D, D, device=DEVICE, dtype=dtype, requires_grad=True)
        bf = torch.randn(D, device=DEVICE, dtype=dtype, requires_grad=True)

        def func(xi, vi, fi, Ui, Wi, Wfi, bfi):
            # Use autograd wrapper directly
            x_o, v_o = leapfrog_fused_autograd(xi, vi, fi, Ui, Wi, 0.1, 1.0, 2, 0, Wfi, bfi, 0.5, 2.0, 1.0)
            return (x_o**2).sum() + (v_o**2).sum()

        torch.autograd.gradcheck(func, (x, v, f, U, W, Wf, bf), eps=1e-5, atol=1e-3, rtol=1e-2)
