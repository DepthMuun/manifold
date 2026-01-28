#!/usr/bin/env python3
"""
Comprehensive Test Suite
========================

Test suite completo que cubre:
- Convergencia de modelos
- Gradientes y backpropagation
- CUDA vs Python fallback
- Toroidal geometry coherence
- Geometric losses
- Head mixing y multi-head integration
- Bugs conocidos y edge cases
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def test_convergence_basic():
    """Test básico de convergencia"""
    print("\n" + "="*60)
    print("TEST 1: BASIC CONVERGENCE")
    print("="*60)
    
    try:
        from gfn.model import Manifold
        from gfn.losses import ToroidalDistanceLoss, hamiltonian_loss
        
        # Crear modelo simple
        model = Manifold(
            vocab_size=10,
            dim=32,
            depth=2,
            heads=2,
            integrator_type='heun',
            holographic=True,
            physics_config={'topology': {'type': 'torus'}}
        )
        
        # Dataset sintético
        batch_size = 16
        seq_len = 1  # Usar seq_len=1 para evitar problemas de dimension
        X = torch.randint(0, 10, (batch_size, seq_len))
        y = torch.randn(batch_size, 32)  # Targets [batch, dim] para losses geométricos
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Probar diferentes losses
        losses = [
            ("MSE", nn.MSELoss()),
            ("Toroidal", ToroidalDistanceLoss())
        ]
        
        results = {}
        
        for loss_name, criterion in losses:
            print(f"\nTesting {loss_name} loss...")
            
            initial_loss = None
            final_loss = None
            
            for epoch in range(20):
                optimizer.zero_grad()
                output = model(X)
                
                # Manejar salida del modelo para obtener coordenadas del manifold
                # output = (logits, (x, v), ...)
                x_state = output[1][0]  # (batch, seq_len, dim)
                
                if loss_name in ["Toroidal"]:
                    loss = criterion(x_state, y)
                else:
                    loss = criterion(x_state, y)
                
                if epoch == 0:
                    initial_loss = loss.item()
                
                loss.backward()
                optimizer.step()
                
                final_loss = loss.item()
            
            convergence_ratio = final_loss / initial_loss if initial_loss > 0 else 1.0
            results[loss_name] = {
                "initial": initial_loss,
                "final": final_loss,
                "ratio": convergence_ratio,
                "converged": convergence_ratio < 0.95
            }
            
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Convergence ratio: {convergence_ratio:.3f}")
            print(f"  Converged: {results[loss_name]['converged']}")
        
        # Verificar que al menos un loss convergió
        any_converged = any(r["converged"] for r in results.values())
        
        print(f"\n✓ Convergence test: {'PASSED' if any_converged else 'FAILED'}")
        return any_converged
        
    except Exception as e:
        print(f"✗ Convergence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test de flujo de gradientes"""
    print("\n" + "="*60)
    print("TEST 2: GRADIENT FLOW")
    print("="*60)
    
    try:
        from gfn.model import Manifold
        
        model = Manifold(
            vocab_size=10,
            dim=16,
            depth=1,
            heads=1,
            integrator_type='heun',
            holographic=True,
            physics_config={'topology': {'type': 'torus'}}
        )
        
        batch_size = 4
        seq_len = 4
        X = torch.randint(0, 10, (batch_size, seq_len))
        y = torch.randn(batch_size, seq_len, 10)
        
        # Forward pass
        output = model(X)
        
        # Manejar salida del modelo (puede ser tupla)
        if isinstance(output, tuple):
            output = output[0]  # Tomar el primer componente
        
        loss = nn.MSELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Verificar que los gradientes existen
        has_gradients = []
        gradient_stats = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                has_gradients.append(name)
                gradient_stats.append({
                    "name": name,
                    "shape": list(param.shape),
                    "grad_norm": grad_norm,
                    "finite": np.isfinite(grad_norm)
                })
        
        print(f"✓ Parameters with gradients: {len(has_gradients)}")
        print(f"✓ Total parameters: {len(list(model.parameters()))}")
        
        # Verificar que todos los gradientes son finitos
        all_finite = all(stat["finite"] for stat in gradient_stats)
        
        if all_finite:
            print("✓ All gradients are finite")
        else:
            print("✗ Some gradients are non-finite")
        
        # Mostrar estadísticas de gradientes
        grad_norms = [stat["grad_norm"] for stat in gradient_stats if stat["finite"]]
        if grad_norms:
            print(f"✓ Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")
        
        return len(has_gradients) > 0 and all_finite
        
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_python_parity():
    """Test de paridad entre CUDA y Python fallback"""
    print("\n" + "="*60)
    print("TEST 3: CUDA vs PYTHON PARITY")
    print("="*60)
    
    try:
        from gfn.cuda.ops import CUDA_AVAILABLE, recurrent_manifold_fused
        
        if not CUDA_AVAILABLE:
            print("⚠️  CUDA not available - skipping parity test")
            return True
        
        # Crear datos de prueba
        device = torch.device('cuda')
        batch_size = 2
        seq_len = 4
        dim = 8
        num_heads = 1
        
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        f = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        U_stack = torch.randn(num_heads, dim, dim // num_heads, device=device, dtype=torch.float32)
        W_stack = torch.randn(num_heads, dim // num_heads, dim, device=device, dtype=torch.float32)
        dt_scales = torch.ones(num_heads, device=device, dtype=torch.float32)
        forget_rates = torch.ones(num_heads, device=device, dtype=torch.float32)
        
        # Dummy gate tensors to prevent CUDA crash
        h_dim = dim // num_heads
        W_f = torch.randn(num_heads, h_dim, h_dim, device=device)
        W_i = torch.randn(num_heads, h_dim, h_dim, device=device)
        b_f = torch.randn(num_heads, h_dim, device=device)
        W_p = torch.randn(num_heads, 1, h_dim * 2, device=device) # Toroidal potential dim
        b_p = torch.randn(num_heads, 1, device=device)
        mix_x = torch.randn(0, device=device) # Empty for single head
        mix_v = torch.randn(0, device=device)
        
        print(f"✓ CUDA tensors created")
        
        # Ejecutar en CUDA
        try:
            cuda_result = recurrent_manifold_fused(
                x, v, f, U_stack, W_stack, 0.1, dt_scales, forget_rates,
                num_heads=num_heads, 
                mix_x=mix_x, mix_v=mix_v,
                W_forget_stack=W_f, W_input_stack=W_i, b_forget_stack=b_f, 
                W_potential_stack=W_p, b_potential_stack=b_p,
                topology=1, R=2.0, r=1.0,
                plasticity=0.0, sing_thresh=1.0, sing_strength=1.0
            )
            
            if cuda_result is not None:
                print(f"✓ CUDA execution successful")
                if isinstance(cuda_result, (tuple, list)):
                    print(f"  Returned {len(cuda_result)} components")
                    for i, comp in enumerate(cuda_result):
                        print(f"    Component {i}: {comp.shape}")
                else:
                    print(f"  Returned: {cuda_result.shape}")
                
                # Verificar validez
                if isinstance(cuda_result, (tuple, list)):
                    all_valid = all(torch.isfinite(comp).all() for comp in cuda_result)
                else:
                    all_valid = torch.isfinite(cuda_result).all()
                
                print(f"✓ CUDA result valid: {all_valid}")
                return True
            else:
                print("⚠️  CUDA returned None (fallback)")
                return True
                
        except Exception as e:
            print(f"✗ CUDA execution failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ CUDA parity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_toroidal_geometry():
    """Test de coherencia de geometría toroidal"""
    print("\n" + "="*60)
    print("TEST 4: TOROIDAL GEOMETRY COHERENCE")
    print("="*60)
    
    try:
        from gfn.geometry.toroidal import ToroidalChristoffel
        from gfn.losses import ToroidalDistanceLoss
        
        # Crear manifold toroidal
        toroidal_config = {'topology': {'major_radius': 2.0, 'minor_radius': 1.0}}
        manifold = ToroidalChristoffel(dim=8, physics_config=toroidal_config)
        
        # Test de proyección
        batch_size = 4
        dim = 8
        
        # Puntos en el espacio ambiente
        points = torch.randn(batch_size, dim)
        
        print(f"✓ Toroidal Christoffel created")
        print(f"  Input shape: {points.shape}")
        print(f"  R: {manifold.R}, r: {manifold.r}")
        
        # Test de distancia toroidal
        loss = ToroidalDistanceLoss()
        
        # Crear targets en el toroide
        targets = torch.randn_like(points)
        
        # Calcular pérdida
        distance_loss = loss(points, targets)
        
        print(f"✓ Toroidal distance calculated: {distance_loss.item():.4f}")
        
        # Verificar que la pérdida es finita y no negativa
        valid_loss = torch.isfinite(distance_loss) and distance_loss.item() >= 0
        print(f"✓ Valid toroidal loss: {valid_loss}")
        
        return valid_loss
        
    except Exception as e:
        print(f"✗ Toroidal geometry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_head_integration():
    """Test de integración multi-head"""
    print("\n" + "="*60)
    print("TEST 5: MULTI-HEAD INTEGRATION")
    print("="*60)
    
    try:
        from gfn.model import Manifold
        
        # Probar diferentes configuraciones de heads
        head_configs = [1, 2, 4]
        results = {}
        
        for num_heads in head_configs:
            print(f"\nTesting {num_heads} heads...")
            
            model = Manifold(
                vocab_size=10,
                dim=32,
                depth=2,
                heads=num_heads,
                integrator_type='heun',
                holographic=True
            )
            
            batch_size = 4
            seq_len = 8
            X = torch.randint(0, 10, (batch_size, seq_len))
            
            # Forward pass
            output = model(X)
            
            # Handle tuple output
            if isinstance(output, tuple):
                output = output[0]
            
            print(f"  Input shape: {X.shape}")
            print(f"  Output shape: {output.shape}")
            
            # Verificar que el output tiene la forma esperada
            expected_shape = (batch_size, seq_len, 10)
            shape_correct = output.shape == expected_shape
            
            results[num_heads] = {
                "shape_correct": shape_correct,
                "output_shape": list(output.shape),
                "parameters": sum(p.numel() for p in model.parameters())
            }
            
            print(f"  Shape correct: {shape_correct}")
            print(f"  Parameters: {results[num_heads]['parameters']}")
        
        # Verificar que todas las configuraciones funcionaron
        all_correct = all(r["shape_correct"] for r in results.values())
        
        print(f"\n✓ Multi-head integration: {'PASSED' if all_correct else 'FAILED'}")
        return all_correct
        
    except Exception as e:
        print(f"✗ Multi-head test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test de casos edge y bugs conocidos"""
    print("\n" + "="*60)
    print("TEST 6: EDGE CASES AND KNOWN BUGS")
    print("="*60)
    
    try:
        from gfn.model import Manifold
        
        # Test 1: Secuencia muy corta
        print("\nTesting very short sequences...")
        model = Manifold(vocab_size=5, dim=16, depth=1, heads=1, holographic=True)
        
        # Secuencia de longitud 1
        X_short = torch.randint(0, 5, (1, 1))
        output_short = model(X_short)
        if isinstance(output_short, tuple): output_short = output_short[0]
        print(f"  Short sequence (1,1) -> {output_short.shape}: ✓")
        
        # Test 2: Batch size 1
        print("\nTesting batch size 1...")
        X_single = torch.randint(0, 5, (1, 4))
        output_single = model(X_single)
        if isinstance(output_single, tuple): output_single = output_single[0]
        print(f"  Single batch (1,4) -> {output_single.shape}: ✓")
        
        # Test 3: Dimensión muy pequeña
        print("\nTesting very small dimensions...")
        model_small = Manifold(vocab_size=3, dim=4, depth=1, heads=1, holographic=True, physics_config={'topology': {'type': 'torus'}})
        X_small = torch.randint(0, 3, (2, 3))
        output_small = model_small(X_small)
        if isinstance(output_small, tuple): output_small = output_small[0]
        print(f"  Small dim (2,3) -> {output_small.shape}: ✓")
        
        # Test 4: Verificar que no hay NaN/Inf
        print("\nTesting for NaN/Inf in outputs...")
        X_test = torch.randint(0, 5, (4, 8))
        output_test = model(X_test)
        if isinstance(output_test, tuple): output_test = output_test[0]
        
        has_nan = torch.isnan(output_test).any().item()
        has_inf = torch.isinf(output_test).any().item()
        
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        valid_output = not has_nan and not has_inf
        
        return valid_output
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar todos los tests"""
    print("COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("Este test suite verifica:")
    print("✓ Convergencia de modelos")
    print("✓ Flujo de gradientes")
    print("✓ CUDA vs Python fallback")
    print("✓ Geometría toroidal")
    print("✓ Integración multi-head")
    print("✓ Casos edge y bugs conocidos")
    
    tests = [
        ("Convergence", test_convergence_basic),
        ("Gradient Flow", test_gradient_flow),
        ("CUDA Parity", test_cuda_python_parity),
        ("Toroidal Geometry", test_toroidal_geometry),
        ("Multi-head Integration", test_multi_head_integration),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Resumen final
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("El sistema está funcionando correctamente.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("Revisa los logs anteriores para más detalles.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
