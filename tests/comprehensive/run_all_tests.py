#!/usr/bin/env python3
"""
Master Test Runner
==================

Ejecuta todos los tests integrados de forma coordinada y genera un reporte final completo.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

def run_test_script(script_path: Path, timeout: int = 300) -> Dict[str, Any]:
    """Ejecutar un script de test individual"""
    print(f"\n{'='*80}")
    print(f"Running: {script_path.name}")
    print('='*80)
    
    start_time = time.time()
    
    try:
        # Ejecutar el script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        
        # Analizar resultado
        success = result.returncode == 0
        
        return {
            "script": script_path.name,
            "success": success,
            "return_code": result.returncode,
            "execution_time_seconds": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": datetime.now().isoformat()
        }
        
    except subprocess.TimeoutExpired:
        return {
            "script": script_path.name,
            "success": False,
            "return_code": -1,
            "execution_time_seconds": timeout,
            "stdout": "",
            "stderr": f"Test timed out after {timeout} seconds",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "script": script_path.name,
            "success": False,
            "return_code": -1,
            "execution_time_seconds": time.time() - start_time,
            "stdout": "",
            "stderr": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_all_tests():
    """Ejecutar todos los tests disponibles"""
    print("="*80)
    print("MASTER TEST RUNNER")
    print("="*80)
    print(f"Python: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Tests a ejecutar (en orden lógico)
    test_scripts = [
        # Tests de CUDA kernels (si están disponibles)
        Path(__file__).parent / "test_cuda_kernels.py",
        
        # Tests de convergencia y estabilidad
        Path(__file__).parent / "test_convergence_stability.py",
        
        # Test suite integrado principal
        Path(__file__).parent / "run_comprehensive_tests.py",
    ]
    
    # Filtrar scripts que existen
    available_tests = [script for script in test_scripts if script.exists()]
    
    if not available_tests:
        print("\n❌ No test scripts found!")
        return
    
    print(f"\nFound {len(available_tests)} test scripts to run:")
    for script in available_tests:
        print(f"  - {script.name}")
    
    # Ejecutar tests
    results = []
    total_start_time = time.time()
    
    for script_path in available_tests:
        result = run_test_script(script_path)
        results.append(result)
        
        # Pequeña pausa entre tests
        time.sleep(1)
    
    total_time = time.time() - total_start_time
    
    # Generar reporte final
    generate_final_report(results, total_time)
    
    return results

def generate_final_report(results: List[Dict[str, Any]], total_time: float):
    """Generar reporte final con todos los resultados"""
    print("\n" + "="*80)
    print("FINAL TEST REPORT")
    print("="*80)
    
    # Estadísticas generales
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"\nSUMMARY:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Total execution time: {total_time:.2f} seconds")
    
    # Detalles por test
    print(f"\nDETAILED RESULTS:")
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"\n{status} - {result['script']}")
        print(f"  Return code: {result['return_code']}")
        print(f"  Execution time: {result['execution_time_seconds']:.2f}s")
        
        if result["stderr"]:
            print(f"  Errors: {result['stderr'][:200]}...")
    
    # Guardar reporte completo
    report_data = {
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "total_execution_time_seconds": total_time,
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }
    
    report_path = Path(__file__).parent / "master_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Generar reporte de texto también
    text_report_path = Path(__file__).parent / "master_test_report.txt"
    with open(text_report_path, 'w') as f:
        f.write("MASTER TEST REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Python: {sys.executable}\n")
        f.write(f"Working directory: {Path.cwd()}\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"  Total tests: {total_tests}\n")
        f.write(f"  Passed: {passed_tests}\n")
        f.write(f"  Failed: {failed_tests}\n")
        f.write(f"  Total execution time: {total_time:.2f} seconds\n\n")
        
        f.write("DETAILED RESULTS:\n")
        for result in results:
            status = "PASSED" if result["success"] else "FAILED"
            f.write(f"\n{status} - {result['script']}\n")
            f.write(f"  Return code: {result['return_code']}\n")
            f.write(f"  Execution time: {result['execution_time_seconds']:.2f}s\n")
            f.write(f"  Timestamp: {result['timestamp']}\n")
            
            if result["stdout"]:
                f.write(f"  Output: {result['stdout'][:500]}...\n")
            
            if result["stderr"]:
                f.write(f"  Errors: {result['stderr'][:500]}...\n")
    
    print(f"📄 Text report saved to: {text_report_path}")
    
    # Código de salida
    if failed_tests > 0:
        print(f"\n❌ {failed_tests} tests failed!")
        sys.exit(1)
    else:
        print(f"\n🎉 All tests passed!")
        sys.exit(0)

def quick_test():
    """Test rápido para verificar que todo está funcionando"""
    print("="*60)
    print("QUICK VERIFICATION TEST")
    print("="*60)
    
    # Verificar que podemos importar los módulos principales
    try:
        from gfn.model import Manifold
        from gfn.cuda.ops import CUDA_AVAILABLE
        print(f"✓ GFN modules imported successfully")
        print(f"  CUDA Available: {CUDA_AVAILABLE}")
    except Exception as e:
        print(f"✗ Failed to import GFN modules: {e}")
        return False
    
    # Verificar que podemos crear un modelo simple
    try:
        model = Manifold(
            vocab_size=2,
            dim=8,
            depth=1,
            heads=1,
            integrator_type='leapfrog',
            holographic=True,
            topology='toroidal'
        )
        print(f"✓ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False
    
    # Verificar forward pass
    try:
        x = torch.randn(2, 4, 8)
        embedded = model.functional_embedding(x)
        output = model(embedded)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    print("\n🎉 Quick verification passed!")
    return True

if __name__ == "__main__":
    # Primero hacer verificación rápida
    if not quick_test():
        print("\n❌ Quick verification failed - aborting full test suite")
        sys.exit(1)
    
    # Luego ejecutar el suite completo
    try:
        results = run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)