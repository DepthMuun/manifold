/**
 * Toroidal Christoffel Fused Kernel
 * ==================================
 * 
 * Dedicated CUDA kernel for computing Christoffel symbols on toroidal manifolds.
 * This kernel implements the metric-derived connection for torus topology.
 * 
 * AUDIT FIX (2026-02-06): Component 2 - Toroidal Geometry in CUDA Fused Mode
 * 
 * Problem: fusion.py was passing dummy zero tensors instead of computing
 * actual toroidal Christoffel symbols, causing complete loss of curvature.
 * 
 * Solution: Dedicated kernel that computes toroidal Christoffel from metric:
 *   ds² = (R + r*cos(θ))² dφ² + r² dθ²
 * 
 * Christoffel symbols (non-zero components):
 *   Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r
 *   Γ^φ_θφ = Γ^φ_φθ = -r*sin(θ) / (R + r*cos(θ))
 * 
 * Author: MiniMax Agent (Audit Implementation)
 * Date: 2026-02-06
 * References: 
 *   - technical_analysis.md: Lines 55-72
 *   - implementation_plan.md: Component 2
 */

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../common/types.cuh"
#include "../../common/device_utils.cuh"
#include "../../common/math_utils.cuh"
#include "../../geometry/christoffel_impl.cuh"

namespace gfn {
namespace cuda {

// ============================================================================
// TOROIDAL CHRISTOFFEL COMPUTATION (Device Function)
// ============================================================================

/**
 * @brief Compute toroidal Christoffel symbols for a single pair (θ, φ)
 * 
 * For toroidal manifold with metric:
 *   ds² = (R + r*cos(θ))² dφ² + r² dθ²
 * 
 * Non-zero Christoffel symbols:
 *   Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r
 *   Γ^φ_θφ = Γ^φ_φθ = -r*sin(θ) / (R + r*cos(θ))
 * 
 * @param theta Position angle θ (poloidal)
 * @param phi Position angle φ (toroidal) - not used in computation but for API consistency
 * @param v_theta Velocity component v^θ
 * @param v_phi Velocity component v^φ
 * @param R Major radius of torus
 * @param r Minor radius of torus
 * @param gamma_theta Output: Christoffel force component Γ(v,v)^θ
 * @param gamma_phi Output: Christoffel force component Γ(v,v)^φ
 */
GFN_DEVICE void toroidal_christoffel_pair(
    scalar_t theta,
    scalar_t phi,  // Unused but kept for API consistency
    scalar_t v_theta,
    scalar_t v_phi,
    scalar_t R,
    scalar_t r,
    scalar_t* gamma_theta,
    scalar_t* gamma_phi
) {
    // Precompute trigonometric functions
    scalar_t sin_theta = sinf(theta);
    scalar_t cos_theta = cosf(theta);
    
    // Compute metric coefficient: g_φφ = (R + r*cos(θ))²
    scalar_t denom = fmax(R + r * cos_theta, CLAMP_MIN_STRONG);
    
    // Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r
    // Christoffel force: -Γ^θ_φφ * v^φ * v^φ
    scalar_t term_theta = denom * sin_theta / (r + EPSILON_SMOOTH);
    *gamma_theta = term_theta * (v_phi * v_phi);
    
    // Γ^φ_θφ = -r*sin(θ) / (R + r*cos(θ))
    // Christoffel force: -2 * Γ^φ_θφ * v^θ * v^φ
    scalar_t term_phi = -(r * sin_theta) / (denom + EPSILON_SMOOTH);
    *gamma_phi = 2.0f * term_phi * v_theta * v_phi;
}

/**
 * @brief Compute toroidal Christoffel force for full dimension vector
 * 
 * Processes dimension in pairs (θ, φ), applies toroidal Christoffel
 * computation to each pair independently.
 * 
 * @param x Position vector [dim] - angles on torus
 * @param v Velocity vector [dim] - tangent space velocities
 * @param dim Dimension (must be even for toroidal pairs)
 * @param R Major radius
 * @param r Minor radius
 * @param gamma Output: Christoffel force [dim]
 */
GFN_DEVICE void toroidal_christoffel_full(
    const scalar_t* x,
    const scalar_t* v,
    int dim,
    scalar_t R,
    scalar_t r,
    scalar_t* gamma
) {
    // Initialize output
    for (int i = 0; i < dim; ++i) {
        gamma[i] = 0.0f;
    }
    
    // Process pairs (θ, φ)
    for (int i = 0; i < dim - 1; i += 2) {
        scalar_t theta = x[i];
        scalar_t phi = x[i + 1];
        scalar_t v_theta = v[i];
        scalar_t v_phi = v[i + 1];
        
        scalar_t g_theta, g_phi;
        toroidal_christoffel_pair(
            theta, phi, v_theta, v_phi,
            R, r,
            &g_theta, &g_phi
        );
        
        gamma[i] = g_theta;
        gamma[i + 1] = g_phi;
    }
    
    // Apply curvature scaling and clamping
    for (int i = 0; i < dim; ++i) {
        gamma[i] = soft_clamp(gamma[i] * TOROIDAL_CURVATURE_SCALE, CURVATURE_CLAMP);
    }
}

// ============================================================================
// TOROIDAL LEAPFROG FUSED KERNEL (Global Function)
// ============================================================================

/**
 * @brief Fused toroidal leapfrog integration kernel
 * 
 * Performs full sequence integration using toroidal Christoffel symbols.
 * This is a specialized version of leapfrog_fused.cu for toroidal topology.
 * 
 * Integration scheme (Kick-Drift-Kick):
 *   1. Compute friction μ(x) from position
 *   2. Compute Christoffel force Γ(v,v)
 *   3. KICK 1: v_half = (v + h*(F - Γ)) / (1 + h*μ)
 *   4. DRIFT: x_new = x + dt * v_half
 *   5. Apply toroidal boundary: x ∈ [0, 2π)
 *   6. Recompute μ(x_new) and Γ(v_half, v_half)
 *   7. KICK 2: v_new = (v_half + h*(F - Γ)) / (1 + h*μ)
 * 
 * @param x Initial position [batch, dim]
 * @param v Initial velocity [batch, dim]
 * @param f Force sequence [batch, seq_len, dim]
 * @param R Major radius
 * @param r Minor radius
 * @param dt Time step
 * @param batch Batch size
 * @param seq_len Sequence length
 * @param dim Dimension
 * @param x_out Output positions [batch, seq_len, dim]
 * @param v_out Output velocities [batch, seq_len, dim]
 */
GFN_GLOBAL void toroidal_leapfrog_fused_kernel(
    const scalar_t* x,
    const scalar_t* v,
    const scalar_t* f,
    const scalar_t* W_forget,   // [dim, feat_dim] or nullptr for DEFAULT_FRICTION
    const scalar_t* b_forget,   // [dim] or nullptr
    scalar_t R,
    scalar_t r,
    scalar_t dt,
    int batch,
    int seq_len,
    int dim,
    scalar_t* x_out,
    scalar_t* v_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch) return;
    
    // Thread-local state
    scalar_t curr_x[256];  // Max dim = 256
    scalar_t curr_v[256];
    scalar_t gamma[256];
    scalar_t friction[256];
    
    // Initialize state
    for (int i = 0; i < dim; ++i) {
        curr_x[i] = x[tid * dim + i];
        curr_v[i] = v[tid * dim + i];
        friction[i] = DEFAULT_FRICTION;  // Default fallback
    }
    
    // Integrate sequence
    for (int t = 0; t < seq_len; ++t) {
        const scalar_t* f_ptr = &f[tid * seq_len * dim + t * dim];
        scalar_t h = dt * 0.5f;  // Half time step
        scalar_t effective_dt = dt;
        
        // BUG-8 FIX: Compute learned friction if weights are provided
        if (W_forget != nullptr && b_forget != nullptr) {
            int feat_dim = 2 * dim; // Toroidal: [sin(x), cos(x)]
            scalar_t features[256]; // Max dim=128 after doubling
            for (int i = 0; i < dim; ++i) {
                features[i] = sinf(curr_x[i]);
                features[dim + i] = cosf(curr_x[i]);
            }
            for (int i = 0; i < dim; ++i) {
                scalar_t gate = b_forget[i];
                for (int j = 0; j < feat_dim; ++j) {
                    gate += W_forget[i * feat_dim + j] * features[j];
                }
                friction[i] = 1.0f / (1.0f + expf(-gate)) * FRICTION_SCALE;
            }
        }
        
        // 1. Compute toroidal Christoffel force at current state
        toroidal_christoffel_full(curr_x, curr_v, dim, R, r, gamma);
        
        // 2. KICK 1 (Half step velocity with implicit friction)
        for (int i = 0; i < dim; ++i) {
            scalar_t numerator = curr_v[i] + h * (f_ptr[i] - gamma[i]);
            scalar_t denominator = 1.0f + h * friction[i];
            curr_v[i] = safe_divide(numerator, denominator, EPSILON_STANDARD);
        }
        
        // 3. DRIFT (Full step position)
        for (int i = 0; i < dim; ++i) {
            curr_x[i] += effective_dt * curr_v[i];
        }
        
        // 4. Apply toroidal boundary conditions: wrap to [0, 2π)
        apply_boundary_vector(curr_x, dim, Topology::TORUS);
        
        // 5. Compute toroidal Christoffel force at new state
        toroidal_christoffel_full(curr_x, curr_v, dim, R, r, gamma);
        
        // 6. KICK 2 (Half step velocity with implicit friction)
        for (int i = 0; i < dim; ++i) {
            scalar_t numerator = curr_v[i] + h * (f_ptr[i] - gamma[i]);
            scalar_t denominator = 1.0f + h * friction[i];
            curr_v[i] = safe_divide(numerator, denominator, EPSILON_STANDARD);
        }
        
        // 7. Store outputs
        for (int i = 0; i < dim; ++i) {
            x_out[tid * seq_len * dim + t * dim + i] = curr_x[i];
            v_out[tid * seq_len * dim + t * dim + i] = curr_v[i];
        }
    }
}

// ============================================================================
// HOST INTERFACE
// ============================================================================

/**
 * @brief Launch toroidal leapfrog kernel
 * 
 * Host-side wrapper for launching toroidal-specific integration.
 * 
 * @param x Initial positions [batch, dim]
 * @param v Initial velocities [batch, dim]
 * @param f Force sequence [batch, seq_len, dim]
 * @param R Major radius
 * @param r Minor radius
 * @param dt Time step
 * @param batch Batch size
 * @param seq_len Sequence length
 * @param dim Dimension
 * @param x_out Output positions [batch, seq_len, dim]
 * @param v_out Output velocities [batch, seq_len, dim]
 * @param stream CUDA stream (optional)
 */
void launch_toroidal_leapfrog_fused(
    const scalar_t* x,
    const scalar_t* v,
    const scalar_t* f,
    const scalar_t* W_forget,
    const scalar_t* b_forget,
    scalar_t R,
    scalar_t r,
    scalar_t dt,
    int batch,
    int seq_len,
    int dim,
    scalar_t* x_out,
    scalar_t* v_out,
    cudaStream_t stream = 0
) {
    int block_size = 256;
    int grid_size = (batch + block_size - 1) / block_size;
    
    toroidal_leapfrog_fused_kernel<<<grid_size, block_size, 0, stream>>>(
        x, v, f, W_forget, b_forget, R, r, dt, batch, seq_len, dim, x_out, v_out
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] toroidal_leapfrog_fused: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace gfn

// ============================================================================
// PyTorch C++ Interface
// ============================================================================

using namespace gfn::cuda;

std::vector<at::Tensor> toroidal_leapfrog_fused(
    at::Tensor x,
    at::Tensor v,
    at::Tensor f,
    float R,
    float r,
    float dt,
    int64_t batch,
    int64_t seq_len,
    int64_t dim,
    at::Tensor W_forget,
    at::Tensor b_forget
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(f.is_cuda(), "f must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, dim]");
    TORCH_CHECK(v.dim() == 2, "v must be 2D [batch, dim]");
    TORCH_CHECK(f.dim() == 3, "f must be 3D [batch, seq_len, dim]");
    TORCH_CHECK(x.size(0) == batch && x.size(1) == dim, "x shape mismatch");
    TORCH_CHECK(v.size(0) == batch && v.size(1) == dim, "v shape mismatch");
    TORCH_CHECK(f.size(0) == batch && f.size(1) == seq_len && f.size(2) == dim, "f shape mismatch");

    auto options = x.options();
    auto x_out = at::empty({batch, seq_len, dim}, options);
    auto v_out = at::empty({batch, seq_len, dim}, options);

    const scalar_t* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<scalar_t>() : nullptr;
    const scalar_t* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<scalar_t>() : nullptr;

    launch_toroidal_leapfrog_fused(
        x.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        f.data_ptr<scalar_t>(),
        W_forget_ptr,
        b_forget_ptr,
        static_cast<scalar_t>(R),
        static_cast<scalar_t>(r),
        static_cast<scalar_t>(dt),
        static_cast<int>(batch),
        static_cast<int>(seq_len),
        static_cast<int>(dim),
        x_out.data_ptr<scalar_t>(),
        v_out.data_ptr<scalar_t>()
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return {x_out, v_out};
}
