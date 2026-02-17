#ifndef GFN_CUDA_TYPES_CUH
#define GFN_CUDA_TYPES_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace gfn {
namespace cuda {

// ============================================================================
// Scalar Types
// ============================================================================

// Template scalar type for flexibility - defaults to float for performance
// Can be instantiated as float or double depending on compilation needs
template<typename T = float>
using scalar_t = T;  // Use float precision to match PyTorch float32 and for better performance on consumer GPUs

// ============================================================================
// Topology Types
// ============================================================================

enum class Topology : int32_t {
    EUCLIDEAN = 0,
    TORUS = 1
};

// ============================================================================
// Physics Constants (Template-based for type safety)
// ============================================================================

// Curvature clamping - REDUCED for stability (matches Python)
template<typename T>
constexpr T CURVATURE_CLAMP = static_cast<T>(3.0);
template<typename T>
constexpr T CURVATURE_CLAMP_MIN = static_cast<T>(-3.0);

// Friction scaling - REDUCED for stability (matches Python)
template<typename T>
constexpr T FRICTION_SCALE = static_cast<T>(0.02);
template<typename T>
constexpr T DEFAULT_FRICTION = static_cast<T>(0.002);

// Mathematical constants
template<typename T>
constexpr T PI = static_cast<T>(3.14159265358979323846);
template<typename T>
constexpr T TWO_PI = static_cast<T>(6.28318530717958647692);

// AUDIT FIX (2026-02-06): Unified epsilon for CUDA/Python parity
// Python: gfn/constants.py::EPSILON_STANDARD = 1e-7
// CUDA: All division safety uses EPSILON_STANDARD = 1e-7
// REMOVED: EPSILON_WEAK (was 1e-6) - inconsistent with Python
template<typename T>
constexpr T EPSILON_STANDARD = static_cast<T>(1e-7);
template<typename T>
constexpr T EPSILON_STRONG = static_cast<T>(1e-7);
template<typename T>
constexpr T EPSILON_SMOOTH = static_cast<T>(1e-7);

// Clamping for division safety
template<typename T>
constexpr T CLAMP_MIN_WEAK = static_cast<T>(1e-7);
template<typename T>
constexpr T CLAMP_MIN_STRONG = static_cast<T>(1e-7);

// Gate biases - MODERATE for stability (matches Python)
template<typename T>
constexpr T GATE_BIAS_OPEN = static_cast<T>(1.0);   // sigmoid(1) ≈ 0.73
template<typename T>
constexpr T GATE_BIAS_CLOSED = static_cast<T>(-3.0); // sigmoid(-3) ≈ 0.05

// Toroidal geometry
template<typename T>
constexpr T TOROIDAL_MAJOR_RADIUS = static_cast<T>(2.0);  // R
template<typename T>
constexpr T TOROIDAL_MINOR_RADIUS = static_cast<T>(1.0);  // r
template<typename T>
constexpr T TOROIDAL_CURVATURE_SCALE = static_cast<T>(0.01);

// Active inference - REDUCED for stability (matches Python)
template<typename T>
constexpr T DEFAULT_PLASTICITY = static_cast<T>(0.02);
template<typename T>
constexpr T SINGULARITY_THRESHOLD = static_cast<T>(0.5);
template<typename T>
constexpr T SINGULARITY_GATE_SLOPE = static_cast<T>(0.5);  // REDUCED from 10.0 for stability
template<typename T>
constexpr T BLACK_HOLE_STRENGTH = static_cast<T>(1.5);

// ============================================================================
// Device Function Attributes
// ============================================================================

#define GFN_DEVICE __device__ __forceinline__
#define GFN_HOST_DEVICE __host__ __device__ __forceinline__
#define GFN_GLOBAL __global__

// ============================================================================
// Grid/Block Configuration
// ============================================================================

// Use CUDA built-in warpSize instead of custom constant for consistency
// constexpr int WARP_SIZE = 32;  // REMOVED: Use CUDA built-in warpSize
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int DEFAULT_BLOCK_SIZE = 256;

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

GFN_HOST_DEVICE int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

inline dim3 get_grid_size(int n, int block_size = DEFAULT_BLOCK_SIZE) {
    return dim3(div_ceil(n, block_size));
}

inline dim3 get_block_size(int block_size = DEFAULT_BLOCK_SIZE) {
    return dim3(block_size);
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_TYPES_CUH
