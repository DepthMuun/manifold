# MANIFOLD Benchmarking & Visualization Suite

## Overview
This directory contains publication-quality benchmarks and visualizations for the Geodesic Flow Networks (MANIFOLD) architecture. These tools provide scientific validation of the model's performance, physical consistency, and competitive advantages.

## Directory Structure

```text
tests/benchmarks/
├── core/             # Standardized performance benchmarks
│   ├── bench_performance.py       # Speed and memory scaling
│   ├── bench_learning_dynamics.py # Convergence and loss analysis
│   ├── bench_sample_efficiency.py # Data efficiency vs Baselines
│   └── ...                        # Specialized feature benchmarks
├── infra/            # Shared benchmarking infrastructure
│   ├── baselines.py               # Competitive model definitions (Transformer, Mamba)
│   ├── utils.py                   # Results logging and performance metrics
│   └── report_generator.py        # Automated HTML report synthesis
├── viz/              # High-fidelity architectural visualizations
│   ├── vis_math_complexity.py    # Math reasoning landscape
│   ├── vis_gfn_superiority.py     # Comparison with Euclidean models
│   ├── vis_manifold.py            # 3D curvature and flow fields
│   └── ...                        # Feature-specific visualizations
└── results/          # Output directory for data and figures
```

## Key Benchmarks

### 🚀 Performance & Scaling (`core/`)
- **`bench_performance.py`**: Comparative analysis of throughput and memory.
- **`bench_scaling.py`**: Empirically proves **O(1)** memory complexity w.r.t sequence length.

### 🔬 Physics & Generalization (`viz/`)
- **`vis_math_complexity.py`**: Validates the model's ability to navigate complex math landscapes via Active Inference.
- **`vis_noether_invariance.py`**: Demonstrates preservation of learned symmetries.

## Quick Start

### Run a Core Benchmark
```bash
python tests/benchmarks/core/bench_performance.py
```

### Generate Visualizations
```bash
python tests/benchmarks/viz/vis_math_complexity.py
```

### Generate Full Report
```bash
python tests/benchmarks/infra/report_generator.py --checkpoint latest.pt
```

## Dashboard & Results
Reports are generated in HTML format with interactive charts. All figures are exported at 300 DPI for publication readiness.

**Last Updated**: 2026-02-17
**Test Suite Version**: 3.0 (Reorganized)
