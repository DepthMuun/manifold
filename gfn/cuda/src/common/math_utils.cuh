#ifndef GFN_CUDA_MATH_UTILS_CUH
#define GFN_CUDA_MATH_UTILS_CUH

#include "types.cuh"
#include "device_utils.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

namespace gfn {
namespace cuda {

// ============================================================================
// Matrix-Vector Operations
// ============================================================================

/**
 * Matrix-vector multiplication: y = A * x
 * A is [m x n], x is [n], y is [m]
 */
template <typename T>
GFN_DEVICE void matvec(
    T* y,
    const T* A,
    const T* x,
    int m,
    int n
) {
    for (int i = 0; i < m; ++i) {
        T sum = static_cast<T>(0);
        for (int j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * Transposed matrix-vector multiplication: y = A^T * x
 * A is [m x n], x is [m], y is [n]
 */
template <typename T>
GFN_DEVICE void matvec_transpose(
    T* y,
    const T* A,
    const T* x,
    int m,
    int n
) {
    for (int j = 0; j < n; ++j) {
        T sum = static_cast<T>(0);
        for (int i = 0; i < m; ++i) {
            sum += A[i * n + j] * x[i];
        }
        y[j] = sum;
    }
}

/**
 * Outer product: C = a ⊗ b (element-wise)
 * Result: C[i] = a[i] * b[i]
 */
template <typename T>
GFN_DEVICE void outer_product_elementwise(
    T* C,
    const T* a,
    const T* b,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        C[i] = a[i] * b[i];
    }
}

// ============================================================================
// Trigonometric Helpers
// ============================================================================

/**
 * Compute sin and cos simultaneously (more efficient).
 */
template <typename T>
GFN_DEVICE void sincos_scalar(T x, T* s, T* c) {
    #ifdef __CUDA_ARCH__
      if constexpr (std::is_same_v<T, double>) {
          sincos(x, s, c);
      } else {
          sincosf(x, s, c);
      }
    #else
      *s = std::sin(x);
      *c = std::cos(x);
    #endif
}

/**
 * Compute Fourier features for toroidal geometry: [sin(x), cos(x)]
 */
template <typename T>
GFN_DEVICE void compute_fourier_features(
    T* features,
    const T* x,
    int dim
) {
    for (int i = 0; i < dim; ++i) {
        T s, c;
        sincos_scalar(x[i], &s, &c);
        features[i] = s;
        features[dim + i] = c;
    }
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_MATH_UTILS_CUH
