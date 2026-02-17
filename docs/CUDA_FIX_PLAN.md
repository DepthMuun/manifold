# CUDA Fix Plan & Logic Audit

## 1. Findings

### A. Critical Logic Inconsistency (Forward vs Backward)
*   **Context**: The project uses a `ChristoffelOperation` which is primarily based on a Low-Rank approximation (`U`, `W` matrices). However, for Toroidal topology, there is an alternative "Analytic" formulation.
*   **The Bug**: 
    *   The **Forward Kernel** (`leapfrog_fused.cu`) uses the Analytic Torus formula ONLY if `V_w == nullptr` (i.e., no singularities). If singularities are present (`V_w != nullptr`), it correctly falls back to the Low-Rank approximation which supports singularity modulation.
    *   The **Backward Kernel** (via `christoffel_device` in `christoffel_impl.cuh`) blindly enters the Analytic Torus branch whenever `topology == TORUS`, **ignoring whether `V_w` is present**.
*   **Consequence**: When training with Singularities on a Torus:
    *   Forward pass calculates `y = LowRank(x) * Singularity(x)`.
    *   Backward pass calculates gradients for `y = AnalyticTorus(x)`.
    *   **Result**: Gradients are completely wrong/uncorrelated with the loss, leading to divergence.

### B. Shared Memory Hazards
*   **Context**: `leapfrog_fused.cu` uses shared memory for efficient reduction and feature storage.
*   **The Bug**: In the Analytic Torus branch of `christoffel_distributed`, the code assumes `v_shared` is available or reuses memory that might conflict with `features_shared` used for friction.
*   **Consequence**: Potential data corruption if the Analytic Torus branch were executed (currently masked because `V_w != nullptr` in production config, but a ticking time bomb).

### C. Compilation Errors
*   **Context**: `leapfrog_backward.cu` had a duplicate parameter `b_forget`.
*   **Status**: Fixed in previous step.

## 2. Implementation Plan

### Step 1: Fix `christoffel_impl.cuh`
Modify `christoffel_device` to align with the Forward kernel logic.
**Change**:
```cpp
// Before
if (topology == Topology::TORUS && x != nullptr) { ... }

// After
if (topology == Topology::TORUS && x != nullptr && V_w == nullptr) { ... }
```
This ensures that if Singularities (`V_w`) are active, we use the Low-Rank path (which supports them).

### Step 2: Audit & Fix `leapfrog_fused.cu`
Review the shared memory allocation and usage in `leapfrog_fused_kernel`.
Ensure that if we ever enter the Analytic Torus branch, we aren't corrupting memory used by Friction or other components.

### Step 3: Verify `velocity_friction_scale`
Ensure consistent application of velocity scaling in friction between Python and CUDA.

### Step 4: Recompile & Verify
1.  Run `compile_fast.bat` (or equivalent command).
2.  Run `tests/cuda/test_christoffel_stage_mismatch.py` (or the new verification script) to confirm gradients match.
