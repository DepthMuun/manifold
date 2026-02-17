#ifndef GFN_CUDA_DEVICE_UTILS_CUH
#define GFN_CUDA_DEVICE_UTILS_CUH

#include "types.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace gfn {
namespace cuda {

// ============================================================================
// Boundary Conditions
// ============================================================================

/**
 * Apply periodic boundary conditions for toroidal topology.
 * Maps coordinates to [0, 2π) range using smooth wrapping.
 */
template <typename T>
GFN_DEVICE T apply_boundary_device(T x, Topology topology) {
    if (topology == Topology::TORUS) {
        T wrapped = atan2(sin(x), cos(x));
        if (wrapped < static_cast<T>(0)) {
            wrapped += static_cast<T>(TWO_PI<T>);
        }
        return wrapped;
    }
    return x;
}

/**
 * Apply boundary conditions to entire vector.
 */
template <typename T>
GFN_DEVICE void apply_boundary_vector(
    T* x,
    int dim,
    Topology topology
) {
    if (topology == Topology::TORUS) {
        for (int i = 0; i < dim; ++i) {
            x[i] = apply_boundary_device<T>(x[i], topology);
        }
    }
}

// ============================================================================
// Safe Mathematical Operations
// ============================================================================

/**
 * Safe division with epsilon protection.
 */
template <typename T>
GFN_DEVICE T safe_divide(
    T numerator,
    T denominator,
    T epsilon = static_cast<T>(EPSILON_STRONG<T>)
) {
    return numerator / (denominator + epsilon);
}

/**
 * Clamp value to range with configurable limits.
 */
template <typename T>
GFN_DEVICE T clamp_value(
    T value,
    T min_val = static_cast<T>(CURVATURE_CLAMP_MIN<T>),
    T max_val = static_cast<T>(CURVATURE_CLAMP<T>)
) {
    return fmin(fmax(value, min_val), max_val);
}

/**
 * Soft clamp using tanh.
 */
template <typename T>
GFN_DEVICE T soft_clamp(
    T value,
    T scale = static_cast<T>(CURVATURE_CLAMP<T>)
) {
    return scale * tanh(value / scale);
}

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * Dot product of two vectors.
 */
template <typename T>
GFN_DEVICE T dot_product(
    const T* a,
    const T* b,
    int dim
) {
    T result = static_cast<T>(0);
    for (int i = 0; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief Compute L2 norm of a vector with numerical stability
 * @param v Input vector
 * @param dim Dimension of vector
 * @return ||v||_2
 */
template <typename T>
GFN_DEVICE T norm(const T* v, int dim) {
    // AUDIT FIX: Use EPSILON_STANDARD for CUDA/Python parity (was EPSILON_WEAK)
    return sqrt(dot_product<T>(v, v, dim) + static_cast<T>(EPSILON_STANDARD<T>));
}

/**
 * Vector addition: c = a + b
 */
template <typename T>
GFN_DEVICE void vector_add(
    T* c,
    const T* a,
    const T* b,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Scaled vector addition: c = a + scale * b
 */
template <typename T>
GFN_DEVICE void vector_add_scaled(
    T* c,
    const T* a,
    T scale,
    const T* b,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        c[i] = a[i] + scale * b[i];
    }
}

/**
 * Vector scaling: b = scale * a
 */
template <typename T>
GFN_DEVICE void vector_scale(
    T* b,
    T scale,
    const T* a,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        b[i] = scale * a[i];
    }
}

/**
 * Copy vector: dst = src
 */
template <typename T>
GFN_DEVICE void vector_copy(
    T* dst,
    const T* src,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        dst[i] = src[i];
    }
}

/**
 * Zero vector: v = 0
 */
template <typename T>
GFN_DEVICE void vector_zero(
    T* v,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        v[i] = static_cast<T>(0);
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * Sigmoid activation.
 */
template <typename T>
GFN_DEVICE T sigmoid(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
}

/**
 * Tanh activation (already in CUDA, but for consistency).
 */
template <typename T>
GFN_DEVICE T tanh_activation(T x) {
    return tanh(x);
}

// ============================================================================
// Warp-Level Primitives
// ============================================================================

/**
 * Warp-level reduction sum.
 */
template <typename T>
GFN_DEVICE T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level reduction sum.
 */
template <typename T>
GFN_DEVICE T block_reduce_sum(T val) {
    __shared__ T shared[32];
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warp_reduce_sum<T>(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : static_cast<T>(0);
    if (wid == 0) val = warp_reduce_sum<T>(val);
    
    return val;
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_DEVICE_UTILS_CUH
