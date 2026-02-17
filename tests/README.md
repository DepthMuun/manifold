# MANIFOLD Professional Test Suite

This directory contains the verification suite for MANIFOLD (formerly GFN). All tests are unified and consistent with the V2 Architecture (Multi-Head Geodesic Flows).

## Structure

### 1. 🏗️ `unit/` (Component Logic)
Tests individual classes in isolation.
*   `test_components.py`: Verifies `Manifold`, `MLayer`, and `RiemannianGating` shapes and forward passes.

### 2. ⚛️ `physics/` (Mathematical Correctness)
Ensures the model adheres to physical laws (Conservation, Gradients).

### 3. 🌐 `geometry/` (Metric Properties)
Verifies Toroidal, Ricci, AdS/CFT and low-rank curvature mathematical consistency.

### 4. 🧠 `functional/` (Behavioral Verification)
**"Does it work as intended?"** - Emergent features like Curiosity exploration, Hysteresis memory, and Time Dilation.

### 5. 🔌 `integration/` (System Stack)
Tests the full system (vNext stack, training loops).

### 6. 📊 `benchmarks/` (Performance & Metrics)
Quantitative analysis of model capabilities and scaling.

## Usage

Run all tests:
```bash
python tests/run_suite.py
```

Run specific benchmark:
```bash
python tests/benchmarks/benchmark_performance.py
```
