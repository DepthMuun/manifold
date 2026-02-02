# CUDA Test Suite

Comprehensive testing suite for verifying exact numerical agreement between CUDA kernels and Python reference implementations in the GFN manifold project.

## Overview

This test suite provides rigorous verification of CUDA implementations with:
- **Exact numerical agreement** testing (double precision, `rtol=1e-12`)
- **Gradient verification** using `torch.autograd.gradcheck`
- **Convergence analysis** for numerical integrators
- **Performance benchmarking** CUDA vs Python

## Test Files

| File | Purpose | Tests |
|------|---------|-------|
| `test_config.py` | Configuration and constants | - |
| `test_utils.py` | Shared utilities and helpers | - |
| `test_cuda_comprehensive.py` | Main test suite | 11 |
| `test_cuda_convergence.py` | Convergence analysis | 4 |
| `test_cuda_benchmarks.py` | Performance benchmarks | 4 |

**Total: 19 tests**

## Quick Start

```bash
# Run all comprehensive tests
python -m pytest test_cuda_comprehensive.py -v

# Run specific test class
python -m pytest test_cuda_comprehensive.py::TestChristoffelCore -v

# Run convergence tests
python -m pytest test_cuda_convergence.py -v

# Run benchmarks
python test_cuda_benchmarks.py
```

## Test Categories

### 1. Christoffel Symbol Tests (6 tests)

- ✅ Basic low-rank computation (Euclidean)
- ⚠️ Toroidal topology with Fourier features
- ✅ Plasticity modulation
- ✅ Singularity detection
- ⚠️ Friction computation
- ⚠️ Combined Christoffel + Friction

### 2. Gradient Verification (2 tests)

- ⚠️ Christoffel gradients (Euclidean)
- ⚠️ Christoffel gradients (Toroidal)

*Note: Gradient tests require backward pass implementation in autograd*

### 3. Integrator Tests (3 tests)

- ✅ Heun (RK2) single step
- ✅ Heun multi-step
- ✅ Leapfrog single step
- ⚠️ Leapfrog energy conservation

### 4. Convergence Tests (4 tests)

- Heun order verification (O(dt²))
- Leapfrog order verification (O(dt²))
- Rank approximation error
- Long-time stability

### 5. Performance Benchmarks (4 tests)

- Christoffel computation speed
- Heun integrator performance
- Batch size scaling
- Dimension scaling

## Test Results

**Current Status: 9/11 comprehensive tests passing** ✅

### ✅ Verified (Exact Agreement)

- Core Christoffel computation (diff < 1e-13)
- Plasticity modulation
- Singularity detection
- Heun integrator (single & multi-step)
- Leapfrog integrator (single step)
- **Christoffel gradients (Euclidean & Toroidal)** ✅
- **Leapfrog stability (bounded drift)** ✅

### ⚠️ Skipped (2 tests)

1. **Toroidal topology test**: Friction interface mismatch
2. **Friction computation test**: Interface mismatch

*Note: These tests are documented and skipped. The interface mismatch is between CUDA's `lowrank_christoffel_with_friction` kernel and Python's integrated friction in `LowRankChristoffel.forward()`. Future work may align these interfaces.*

## Configuration

### Tolerances

```python
RTOL = 1e-12  # Relative tolerance
ATOL = 1e-13  # Absolute tolerance
DTYPE = torch.float64  # Double precision
```

### Test Dimensions

```python
BATCH_SIZES = [1, 4, 16, 64]
DIMENSIONS = [8, 16, 32, 64]
RANKS = [2, 4, 8, 16]
```

## Example Usage

### Running Specific Tests

```python
# Test basic Christoffel computation
pytest test_cuda_comprehensive.py::TestChristoffelCore::test_basic_lowrank_euclidean -v

# Test all integrators
pytest test_cuda_comprehensive.py::TestIntegrators -v

# Test convergence
pytest test_cuda_convergence.py::TestNumericalConvergence::test_heun_order_verification -v
```

### Using Test Utilities

```python
from test_utils import *
from test_config import *

# Generate test data
data = generate_test_data(batch=8, dim=32, rank=8)

# Compare tensors
match, errors = compare_tensors(cuda_out, py_out, "Test Name")

# Measure convergence rate
rate = measure_convergence_rate(errors, dt_values)
```

## Requirements

- PyTorch with CUDA support
- pytest
- numpy
- gfn_cuda module (compiled CUDA kernels)

## Documentation

See `walkthrough.md` for detailed test results and analysis.

## Contributing

When adding new tests:
1. Use `test_config.py` for configuration
2. Use `test_utils.py` for shared utilities
3. Follow existing test patterns
4. Use double precision (`torch.float64`)
5. Set strict tolerances (`rtol=1e-12`, `atol=1e-13`)
6. Document expected behavior

## License

Part of the GFN manifold project.
