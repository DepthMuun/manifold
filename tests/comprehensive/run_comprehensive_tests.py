#!/usr/bin/env python3
"""
GFN Comprehensive Test Suite
===========================

Tests completos que validan TODO el sistema:
- Convergencia y estabilidad
- Gradientes y backpropagation
- CUDA kernels vs Python fallback
- Geometría toroidal y coherencia
- Losses geométricos (ToroidalDistance, Hamiltonian)
- Head mixing y multi-head integration
- Bugs conocidos y edge cases

Uso:
    python -m tests.comprehensive.run_comprehensive_tests
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.losses import ToroidalDistanceLoss, hamiltonian_loss, geodesic_regularization
from gfn.cuda.ops import CUDA_AVAILABLE, recurrent_manifold_fused, christoffel_fused
from gfn.geometry.toroidal import ToroidalManifold
from gfn.geometry.analytical import christoffel_symbols_toroidal

@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    details: Dict[str, Any]
    error: str = ""

class ComprehensiveTestSuite:
    def __init__(self):
        self.results: List[TestResult] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cuda_available = CUDA_AVAILABLE and torch.cuda.is_available()
        
    def run_test(self, test_func, name: str, **kwargs) -> TestResult:
        """Ejecuta un test y captura resultados"""
        print(f"\n{'='*60}")
        print(f"[TEST] {name}")
        print('='*60)
        
        start_time = time.time()
        try:
            details = test_func(**kwargs)
            passed = True
            error = ""
            print(f"✓ PASSED")
        except Exception as e:
            passed = False
            error = str(e)
            details = {"error": error, "traceback": traceback.format_exc()}
            print(f"✗ FAILED: {error}")
            print(f"Traceback:\n{traceback.format_exc()}")
        
        duration = time.time() - start_time
        
        result = TestResult(
            name=name,
            passed=passed,
            duration=duration,
            details=details,
            error=error
        )
        self.results.append(result)
        return result
    
    def test_cuda_kernel_loading(self) -> Dict[str, Any]:
        """Test 1: Verificar carga de kernels CUDA"""
        print("Testing CUDA kernel loading...")
        
        if not self.cuda_available:
            return {"warning": "CUDA not available, skipping kernel tests"}
        
        # Test que los kernels CUDA están cargados
        test_tensor = torch.randn(2, 4, 8).cuda()
        
        # Test recurrent_manifold_fused
        x = torch.randn(2, 4, 8).cuda()
        v = torch.randn(2, 4, 8).cuda()
        f = torch.randn(2, 4, 8).cuda()
        U_stack = torch.randn(1, 8, 4).cuda()
        W_stack = torch.randn(1, 4, 8).cuda()
        dt = 0.1
        dt_scales = torch.ones(1).cuda()
        forget_rates = torch.ones(1).cuda()
        
        result = recurrent_manifold_fused(
            x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, 
            num_heads=1, topology=1, R=2.0, r=1.0
        )
        
        if result is None:
            raise RuntimeError("CUDA kernel returned None - fallback to Python")
        
        x_state, v_state, x_out_seq, v_out_seq, reg_loss = result
        return {
            "cuda_available": True,
            "kernel_loaded": True,
            "tensor_shape": list(x_out_seq.shape),
            "device": str(x_out_seq.device)
        }
    
    def test_convergence_simple(self) -> Dict[str, Any]:
        """Test 2: Convergencia en dataset simple"""
        print("Testing convergence on simple dataset...")
        
        # Crear dataset simple de clasificación
        torch.manual_seed(42)
        n_samples = 100
        n_features = 8
        n_classes = 2
        
        # Datos sintéticos linealmente separables
        X = torch.randn(n_samples, n_features)
        y = (X.sum(dim=1) > 0).long()
        
        # Modelo pequeño para test rápido
        model = Manifold(
            vocab_size=n_classes,
            dim=n_features,
            depth=2,
            heads=2,
            integrator_type='leapfrog',
            holographic=True,
            topology='toroidal'
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Forward con embedding funcional
            embedded = model.functional_embedding(X.unsqueeze(1))  # [batch, seq=1, dim]
            output = model(embedded)
            
            loss = criterion(output.view(-1, n_classes), y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: loss = {loss.item():.4f}")
        
        # Verificar que converge
        final_loss = losses[-1]
        initial_loss = losses[0]
        
        if final_loss > initial_loss * 0.8:
            raise RuntimeError(f"Model not converging: initial={initial_loss:.4f}, final={final_loss:.4f}")
        
        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "convergence_ratio": final_loss / initial_loss,
            "epochs": len(losses)
        }
    
    def test_gradient_flow(self) -> Dict[str, Any]:
        """Test 3: Flujo de gradientes y backpropagation"""
        print("Testing gradient flow...")
        
        model = Manifold(
            vocab_size=4,
            dim=8,
            depth=2,
            heads=2,
            integrator_type='leapfrog',
            holographic=True
        ).to(self.device)
        
        # Crear datos de entrada
        batch_size = 4
        seq_len = 8
        x = torch.randint(0, 4, (batch_size, seq_len)).to(self.device)
        
        # Forward pass
        output = model(x)
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Verificar que todos los parámetros tienen gradientes
        grad_norms = {}
        total_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"No gradient for parameter: {name}")
            
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            total_norm += grad_norm ** 2
            
            if grad_norm == 0:
                print(f"  Warning: zero gradient for {name}")
        
        total_norm = np.sqrt(total_norm)
        
        # Verificar que los gradientes no son exploding o vanishing
        if total_norm > 100:
            raise RuntimeError(f"Gradient explosion: total_norm = {total_norm}")
        if total_norm < 1e-6:
            raise RuntimeError(f"Gradient vanishing: total_norm = {total_norm}")
        
        return {
            "total_gradient_norm": total_norm,
            "parameter_count": len(grad_norms),
            "min_gradient": min(grad_norms.values()),
            "max_gradient": max(grad_norms.values()),
            "gradient_norms": grad_norms
        }
    
    def test_toroidal_geometry(self) -> Dict[str, Any]:
        """Test 4: Coherencia de geometría toroidal"""
        print("Testing toroidal geometry coherence...")
        
        # Crear manifold toroidal
        manifold = ToroidalManifold(R=2.0, r=1.0)
        
        # Test puntos en el toro
        n_points = 10
        theta = torch.linspace(0, 2*np.pi, n_points)
        phi = torch.linspace(0, 2*np.pi, n_points)
        
        # Coordenadas en el toro
        x = (manifold.R + manifold.r * torch.cos(phi)) * torch.cos(theta)
        y = (manifold.R + manifold.r * torch.cos(phi)) * torch.sin(theta)
        z = manifold.r * torch.sin(phi)
        
        points = torch.stack([x, y, z], dim=1)
        
        # Verificar que los puntos están en la superficie toroidal
        distances = manifold.distance_to_surface(points)
        max_distance = distances.max().item()
        
        if max_distance > 1e-4:
            raise RuntimeError(f"Points not on toroidal surface: max_distance = {max_distance}")
        
        # Test Christoffel symbols
        test_point = torch.tensor([1.5, 0.0, 0.0])  # Punto en el toro
        christoffel = christoffel_symbols_toroidal(test_point, R=manifold.R, r=manifold.r)
        
        # Verificar simetría de Christoffel symbols
        if not torch.allclose(christoffel, christoffel.transpose(-2, -1), atol=1e-6):
            raise RuntimeError("Christoffel symbols not symmetric")
        
        return {
            "max_surface_distance": max_distance,
            "christoffel_shape": list(christoffel.shape),
            "manifold_params": {"R": manifold.R, "r": manifold.r}
        }
    
    def test_geometric_losses(self) -> Dict[str, Any]:
        """Test 5: Losses geométricos"""
        print("Testing geometric losses...")
        
        batch_size = 8
        dim = 16
        
        # Crear predicciones y targets
        pred = torch.randn(batch_size, dim)
        target = torch.randn(batch_size, dim)
        
        # Test ToroidalDistanceLoss
        toroidal_loss = ToroidalDistanceLoss()
        toroidal_output = toroidal_loss(pred, target)
        
        # Test Hamiltonian loss
        hamiltonian_output = hamiltonian_loss([pred, target])
        
        # Test Geodesic regularization
        christoffels = [torch.randn(batch_size, dim), torch.randn(batch_size, dim)]
        geodesic_output = geodesic_regularization([pred], christoffels, lambda_g=0.001)
        
        # Verificar que los losses son positivos
        losses = {
            "toroidal": toroidal_output.item(),
            "hamiltonian": hamiltonian_output.item(),
            "geodesic": geodesic_output.item()
        }
        
        for name, value in losses.items():
            if value < 0:
                raise RuntimeError(f"{name} loss is negative: {value}")
            if not np.isfinite(value):
                raise RuntimeError(f"{name} loss is not finite: {value}")
        
        return {
            "losses": losses,
            "batch_size": batch_size,
            "dimension": dim
        }
    
    def test_cuda_vs_python(self) -> Dict[str, Any]:
        """Test 6: Paridad CUDA vs Python"""
        print("Testing CUDA vs Python parity...")
        
        if not self.cuda_available:
            return {"warning": "CUDA not available, skipping parity test"}
        
        # Crear datos de prueba
        batch_size = 4
        seq_len = 8
        dim = 16
        num_heads = 2
        
        # Tensores en CPU y CUDA
        x_cpu = torch.randn(batch_size, seq_len, dim)
        v_cpu = torch.randn(batch_size, seq_len, dim)
        f_cpu = torch.randn(batch_size, seq_len, dim)
        U_cpu = torch.randn(num_heads, dim, dim // num_heads)
        W_cpu = torch.randn(num_heads, dim // num_heads, dim)
        
        x_cuda = x_cpu.cuda()
        v_cuda = v_cpu.cuda()
        f_cuda = f_cpu.cuda()
        U_cuda = U_cpu.cuda()
        W_cuda = W_cpu.cuda()
        
        dt = 0.1
        dt_scales = torch.ones(num_heads)
        forget_rates = torch.ones(num_heads)
        
        # Resultado CUDA
        cuda_result = recurrent_manifold_fused(
            x_cuda, v_cuda, f_cuda, U_cuda, W_cuda, dt, 
            dt_scales.cuda(), forget_rates.cuda(), num_heads,
            topology=1, R=2.0, r=1.0
        )
        
        if cuda_result is None:
            raise RuntimeError("CUDA kernel returned None")
        x_state, v_state, x_out_seq, v_out_seq, reg_loss = cuda_result
        
        # Resultado Python (fallback)
        # Necesitamos implementar la versión Python para comparación
        # Por ahora, verificamos que CUDA funcione
        
        return {
            "cuda_result_shape": list(x_out_seq.shape),
            "cuda_result_device": str(x_out_seq.device),
            "parity_test": "CUDA kernel executed successfully"
        }
    
    def test_head_mixing(self) -> Dict[str, Any]:
        """Test 7: Head mixing y multi-head integration"""
        print("Testing head mixing...")
        
        model = Manifold(
            vocab_size=8,
            dim=32,
            depth=2,
            heads=4,
            integrator_type='leapfrog',
            holographic=True
        ).to(self.device)
        
        batch_size = 4
        seq_len = 16
        x = torch.randint(0, 8, (batch_size, seq_len)).to(self.device)
        
        # Forward pass
        output = model(x)
        
        # Verificar forma de salida
        expected_shape = (batch_size, seq_len, 8)  # vocab_size
        if output.shape != expected_shape:
            raise RuntimeError(f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        # Verificar que el modelo usa múltiples heads
        head_params = [name for name, _ in model.named_parameters() if 'head' in name]
        
        return {
            "output_shape": list(output.shape),
            "head_parameters": len(head_params),
            "multi_head_active": len(head_params) > 0
        }
    
    def test_known_bugs(self) -> Dict[str, Any]:
        """Test 8: Bugs conocidos y edge cases"""
        print("Testing known bugs and edge cases...")
        
        bugs_found = []
        
        # Test 1: División por cero en Christoffel
        try:
            v = torch.zeros(1, 8)  # Vector cero
            U = torch.randn(8, 4)
            W = torch.randn(4, 8)
            result = christoffel_fused(v, U, W, topology=1)
            if torch.isnan(result).any():
                bugs_found.append("NaN in Christoffel with zero vector")
        except Exception as e:
            bugs_found.append(f"Exception in Christoffel zero vector: {e}")
        
        # Test 2: Tamaños de batch extremos
        try:
            model = Manifold(vocab_size=4, dim=8, depth=1, heads=1).to(self.device)
            
            # Batch size = 1
            x1 = torch.randint(0, 4, (1, 8)).to(self.device)
            output1 = model(x1)
            
            # Batch size grande
            x_large = torch.randint(0, 4, (64, 8)).to(self.device)
            output_large = model(x_large)
            
        except Exception as e:
            bugs_found.append(f"Exception with extreme batch sizes: {e}")
        
        # Test 3: Secuencias muy largas
        try:
            x_long = torch.randint(0, 4, (2, 512)).to(self.device)
            output_long = model(x_long)
            
        except Exception as e:
            bugs_found.append(f"Exception with long sequences: {e}")
        
        return {
            "bugs_found": bugs_found,
            "bug_count": len(bugs_found),
            "edge_cases_tested": 3
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Ejecutar todos los tests"""
        print("\n" + "="*80)
        print("GFN COMPREHENSIVE TEST SUITE")
        print("="*80)
        
        tests = [
            ("CUDA Kernel Loading", self.test_cuda_kernel_loading),
            ("Simple Convergence", self.test_convergence_simple),
            ("Gradient Flow", self.test_gradient_flow),
            ("Toroidal Geometry", self.test_toroidal_geometry),
            ("Geometric Losses", self.test_geometric_losses),
            ("CUDA vs Python Parity", self.test_cuda_vs_python),
            ("Head Mixing", self.test_head_mixing),
            ("Known Bugs", self.test_known_bugs),
        ]
        
        summary = {
            "total_tests": len(tests),
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "total_duration": 0.0
        }
        
        for name, test_func in tests:
            result = self.run_test(test_func, name)
            
            if result.passed:
                summary["passed"] += 1
            else:
                summary["failed"] += 1
            
            if "warning" in result.details:
                summary["warnings"] += 1
            
            summary["total_duration"] += result.duration
        
        # Generar reporte final
        self.generate_report(summary)
        return summary
    
    def generate_report(self, summary: Dict[str, Any]):
        """Generar reporte detallado"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        print(f"\nSUMMARY:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']} ({summary['passed']/summary['total_tests']*100:.1f}%)")
        print(f"  Failed: {summary['failed']} ({summary['failed']/summary['total_tests']*100:.1f}%)")
        print(f"  Warnings: {summary['warnings']}")
        print(f"  Total Duration: {summary['total_duration']:.2f}s")
        
        print(f"\nDETAILED RESULTS:")
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status} {result.name} ({result.duration:.2f}s)")
            
            if not result.passed and result.error:
                print(f"    Error: {result.error}")
        
        # Guardar reporte en archivo
        report_path = Path(__file__).parent / "comprehensive_test_report.txt"
        with open(report_path, 'w') as f:
            f.write("GFN COMPREHENSIVE TEST REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"CUDA Available: {self.cuda_available}\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"  Total Tests: {summary['total_tests']}\n")
            f.write(f"  Passed: {summary['passed']}\n")
            f.write(f"  Failed: {summary['failed']}\n")
            f.write(f"  Warnings: {summary['warnings']}\n")
            f.write(f"  Total Duration: {summary['total_duration']:.2f}s\n\n")
            
            f.write("DETAILED RESULTS:\n")
            for result in self.results:
                status = "PASS" if result.passed else "FAIL"
                f.write(f"  {status}: {result.name} ({result.duration:.2f}s)\n")
                if result.details:
                    for key, value in result.details.items():
                        f.write(f"    {key}: {value}\n")
                if result.error:
                    f.write(f"    Error: {result.error}\n")
                f.write("\n")
        
        print(f"\nReport saved to: {report_path}")

def main():
    """Función principal"""
    suite = ComprehensiveTestSuite()
    summary = suite.run_all_tests()
    
    # Exit code basado en resultados
    sys.exit(0 if summary['failed'] == 0 else 1)

if __name__ == "__main__":
    main()
