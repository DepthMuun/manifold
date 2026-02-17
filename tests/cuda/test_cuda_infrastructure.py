"""
GFN CUDA Infrastructure Suite
=============================
Consolidated suite for:
- CUDA availability and version checks
- Kernel loading and compilation verification
- Basic model forward pass (Smoke tests)

Usage: pytest tests/cuda/test_cuda_infrastructure.py
"""

import torch
import pytest
import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestCUDAInfrastructure:
    """Diagnostic and loading tests."""

    def test_cuda_available(self):
        """Verify that PyTorch can see CUDA hardware."""
        assert torch.cuda.is_available(), "CUDA hardware not detected by PyTorch"
        print(f"\n[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")

    def test_kernel_import(self):
        """Verify that CUDA operators can be imported."""
        try:
            from gfn.cuda.ops import christoffel_fused, leapfrog_fused, CUDA_AVAILABLE
            assert CUDA_AVAILABLE, "CUDA_AVAILABLE flag is False after import"
            assert christoffel_fused is not None
            assert leapfrog_fused is not None
        except ImportError as e:
            pytest.fail(f"Could not import CUDA operators: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_kernel_smoke_run(self):
        """Run a minimal kernel execution to verify binding."""
        from gfn.cuda.ops import christoffel_fused
        
        batch, dim, rank = 2, 32, 4
        v = torch.randn(batch, dim, device='cuda', dtype=torch.float32)
        U = torch.randn(dim, rank, device='cuda', dtype=torch.float32)
        W = torch.randn(dim, rank, device='cuda', dtype=torch.float32)
        
        gamma = christoffel_fused(v, U, W)
        assert gamma.shape == (batch, dim)
        assert not torch.isnan(gamma).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_forward_smoke(self):
        """Verify that the full Manifold model can run with CUDA path."""
        from gfn.model import Manifold
        
        device = 'cuda'
        model = Manifold(
            vocab_size=10,
            dim=64,
            depth=1,
            heads=1,
            rank=8,
            integrator_type='leapfrog',
            use_scan=False
        ).to(device)
        
        inputs = torch.randint(0, 10, (2, 5)).to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(inputs, collect_christ=False)
            logits = outputs[0]
            
        assert logits.shape == (2, 5, 10)
        assert not torch.isnan(logits).any()
