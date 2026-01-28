#!/usr/bin/env python3
"""
Convergence and Stability Test Suite
===================================

Tests de convergencia y estabilidad del modelo GFN con diferentes configuraciones.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.losses import ToroidalDistanceLoss, HamiltonianLoss, GeodesicLoss
from gfn.geometry.toroidal import ToroidalManifold

def generate_synthetic_dataset(n_samples: int = 100, n_features: int = 8, 
                              n_classes: int = 2, noise: float = 0.1, 
                              pattern: str = "linear") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generar dataset sintético para pruebas"""
    torch.manual_seed(42)
    
    if pattern == "linear":
        # Datos linealmente separables con ruido
        X = torch.randn(n_samples, n_features)
        weights = torch.randn(n_features)
        logits = X @ weights + torch.randn(n_samples) * noise
        y = (logits > 0).long()
        
    elif pattern == "circular":
        # Datos circulares (para test toroidal)
        X = torch.randn(n_samples, n_features)
        # Crear patrón circular usando coordenadas polares
        radius = torch.norm(X[:, :2], dim=1)
        angle = torch.atan2(X[:, 1], X[:, 0])
        y = ((radius > 1.0) & (angle > 0)).long()
        
    elif pattern == "xor":
        # XOR pattern
        X = torch.randn(n_samples, n_features)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).long()
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return X, y

def test_convergence_basic():
    """Test 1: Convergencia básica con diferentes configuraciones"""
    print("\n[TEST] Basic Convergence")
    print("-" * 40)
    
    configs = [
        {"name": "Small Linear", "n_samples": 50, "n_features": 4, "depth": 1, "heads": 1, "pattern": "linear"},
        {"name": "Medium Circular", "n_samples": 100, "n_features": 8, "depth": 2, "heads": 2, "pattern": "circular"},
        {"name": "Large XOR", "n_samples": 200, "n_features": 16, "depth": 3, "heads": 4, "pattern": "xor"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        # Generar datos
        X, y = generate_synthetic_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            pattern=config["pattern"]
        )
        
        # Crear modelo
        model = Manifold(
            vocab_size=2,
            dim=config["n_features"],
            depth=config["depth"],
            heads=config["heads"],
            integrator_type='leapfrog',
            holographic=True,
            topology='toroidal'
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Entrenamiento
        losses = []
        n_epochs = 30
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward con embedding funcional
            embedded = model.functional_embedding(X.unsqueeze(1))
            output = model(embedded)
            
            loss = criterion(output.view(-1, 2), y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Análisis de convergencia
        initial_loss = losses[0]
        final_loss = losses[-1]
        convergence_ratio = final_loss / initial_loss
        
        # Verificar que converge
        converged = convergence_ratio < 0.8
        
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Convergence ratio: {convergence_ratio:.3f}")
        print(f"  Converged: {converged}")
        
        results.append({
            "config": config["name"],
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "convergence_ratio": convergence_ratio,
            "converged": converged,
            "losses": losses
        })
    
    # Resumen
    converged_count = sum(1 for r in results if r["converged"])
    
    return {
        "status": "passed" if converged_count == len(configs) else "failed",
        "configs_tested": len(configs),
        "converged_count": converged_count,
        "results": results
    }

def test_stability_long_training():
    """Test 2: Estabilidad en entrenamiento largo"""
    print("\n[TEST] Long Training Stability")
    print("-" * 40)
    
    # Configuración para test largo
    n_samples = 100
    n_features = 8
    n_epochs = 200
    
    # Generar datos
    X, y = generate_synthetic_dataset(n_samples, n_features, pattern="linear")
    
    # Crear modelo
    model = Manifold(
        vocab_size=2,
        dim=n_features,
        depth=2,
        heads=2,
        integrator_type='leapfrog',
        holographic=True,
        topology='toroidal'
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Entrenamiento largo
    losses = []
    gradient_norms = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        embedded = model.functional_embedding(X.unsqueeze(1))
        output = model(embedded)
        
        loss = criterion(output.view(-1, 2), y)
        loss.backward()
        
        # Calcular norma de gradiente
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)
        
        optimizer.step()
        
        losses.append(loss.item())
        gradient_norms.append(grad_norm)
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
    
    # Análisis de estabilidad
    loss_std = np.std(losses[-50:])  # Desviación estándar de últimas 50 épocas
    grad_norm_mean = np.mean(gradient_norms[-50:])
    
    # Verificar que no hay divergencia
    max_loss = max(losses)
    min_loss = min(losses)
    
    stable = True
    issues = []
    
    if loss_std > 0.1:
        issues.append(f"High loss variance: {loss_std:.4f}")
        stable = False
    
    if max_loss > 10 * min_loss:
        issues.append(f"Large loss variation: {max_loss:.4f} vs {min_loss:.4f}")
        stable = False
    
    if grad_norm_mean > 100:
        issues.append(f"High gradient norm: {grad_norm_mean:.4f}")
        stable = False
    
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss std (last 50): {loss_std:.4f}")
    print(f"  Gradient norm mean (last 50): {grad_norm_mean:.4f}")
    print(f"  Stable: {stable}")
    
    if issues:
        print(f"  Issues: {', '.join(issues)}")
    
    return {
        "status": "passed" if stable else "failed",
        "final_loss": losses[-1],
        "loss_std": loss_std,
        "grad_norm_mean": grad_norm_mean,
        "max_loss": max_loss,
        "min_loss": min_loss,
        "stable": stable,
        "issues": issues
    }

def test_geometric_loss_convergence():
    """Test 3: Convergencia con losses geométricos"""
    print("\n[TEST] Geometric Loss Convergence")
    print("-" * 40)
    
    # Generar datos
    X, y = generate_synthetic_dataset(n_samples=100, n_features=8, pattern="circular")
    
    losses_to_test = [
        {"name": "CrossEntropy", "loss": nn.CrossEntropyLoss(), "type": "standard"},
        {"name": "ToroidalDistance", "loss": ToroidalDistanceLoss(R=2.0, r=1.0), "type": "geometric"},
        {"name": "Hamiltonian", "loss": HamiltonianLoss(), "type": "geometric"},
        {"name": "Geodesic", "loss": GeodesicLoss(), "type": "geometric"},
    ]
    
    results = []
    
    for loss_config in losses_to_test:
        print(f"\nTesting {loss_config['name']} loss...")
        
        # Crear modelo
        model = Manifold(
            vocab_size=2,
            dim=8,
            depth=2,
            heads=2,
            integrator_type='leapfrog',
            holographic=True,
            topology='toroidal'
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = loss_config["loss"]
        
        # Entrenamiento
        losses = []
        n_epochs = 50
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            embedded = model.functional_embedding(X.unsqueeze(1))
            output = model(embedded)
            
            # Adaptar targets según el tipo de loss
            if loss_config["type"] == "geometric":
                # Para losses geométricos, usar representación continua
                targets = torch.randn_like(output)
                loss = criterion(output, targets)
            else:
                loss = criterion(output.view(-1, 2), y)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Análisis
        initial_loss = losses[0]
        final_loss = losses[-1]
        convergence_ratio = final_loss / initial_loss
        
        # Verificar que el loss es finito y positivo
        valid_loss = np.isfinite(final_loss) and final_loss >= 0
        
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Convergence ratio: {convergence_ratio:.3f}")
        print(f"  Valid loss: {valid_loss}")
        
        results.append({
            "loss_name": loss_config["name"],
            "loss_type": loss_config["type"],
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "convergence_ratio": convergence_ratio,
            "valid_loss": valid_loss,
            "losses": losses
        })
    
    # Verificar que todos los losses son válidos
    all_valid = all(r["valid_loss"] for r in results)
    
    return {
        "status": "passed" if all_valid else "failed",
        "losses_tested": len(losses_to_test),
        "all_valid": all_valid,
        "results": results
    }

def run_convergence_suite():
    """Ejecutar suite completa de convergencia y estabilidad"""
    print("="*60)
    print("CONVERGENCE AND STABILITY TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Convergence", test_convergence_basic),
        ("Long Training Stability", test_stability_long_training),
        ("Geometric Loss Convergence", test_geometric_loss_convergence),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            
            result = test_func()
            results[test_name] = result
            
            status = result.get("status", "unknown")
            if status == "passed":
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            results[test_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Resumen final
    print("\n" + "="*60)
    print("CONVERGENCE SUITE SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    total = len(tests)
    
    print(f"Tests Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = result.get("status", "unknown")
        if status == "passed":
            print(f"✓ {test_name}")
        elif status == "failed":
            print(f"✗ {test_name}")
        else:
            print(f"? {test_name} (error)")
    
    # Guardar resultados
    import json
    report_path = Path(__file__).parent / "convergence_stability_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return results

if __name__ == "__main__":
    import sys
    results = run_convergence_suite()
    
    # Exit con código apropiado
    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    total = len(results)
    
    if passed == total:
        print("\n🎉 All convergence tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} convergence tests failed")
        sys.exit(1)