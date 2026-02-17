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
    float theta,
    float phi,  // Unused but kept for API consistency
    float v_theta,
    float v_phi,
    float R,
    float r,
    float* gamma_theta,
    float* gamma_phi
) {
    // Precompute trigonometric functions
    float sin_theta = sinf(theta);
    float cos_theta = cosf(theta);
    
    // Compute metric coefficient: g_φφ = (R + r*cos(θ))²
    float denom = fmaxf(R + r * cos_theta, CLAMP_MIN_STRONG<float>);
    
    // Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r
    // Christoffel force: -Γ^θ_φφ * v^φ * v^φ
    float term_theta = denom * sin_theta / (r + EPSILON_SMOOTH<float>);
    *gamma_theta = term_theta * (v_phi * v_phi);
    
    // Γ^φ_θφ = -r*sin(θ) / (R + r*cos(θ))
    // Christoffel force: -2 * Γ^φ_θφ * v^θ * v^φ
    float term_phi = -(r * sin_theta) / (denom + EPSILON_SMOOTH<float>);
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
    const float* x,
    const float* v,
    int dim,
    float R,
    float r,
    float* gamma
) {
    // Initialize output
    for (int i = 0; i < dim; ++i) {
        gamma[i] = 0.0f;
    }
    
    // Process pairs (θ, φ)
    for (int i = 0; i < dim - 1; i += 2) {
        float theta = x[i];
        float phi = x[i + 1];
        float v_theta = v[i];
        float v_phi = v[i + 1];
        
        float g_theta, g_phi;
        toroidal_christoffel_pair(
            theta, phi, v_theta, v_phi,
            R, r,
            &g_theta, &g_phi
        );
        
        gamma[i] = g_theta;
        gamma[i + 1] = g_phi;
    }
    
    // Apply curvature scaling and clamping (PARITY FIX: use CURVATURE_CLAMP)
    for (int i = 0; i < dim; ++i) {
        gamma[i] = fminf(CURVATURE_CLAMP<float>, fmaxf(-CURVATURE_CLAMP<float>, gamma[i]));
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
    const float* x,
    const float* v,
    const float* f,
    const float* W_forget,   // [dim, feat_dim] or nullptr for DEFAULT_FRICTION
    const float* b_forget,   // [dim] or nullptr
    float R,
    float r,
    float dt,
    int batch,
    int seq_len,
    int dim,
    float* x_out,
    float* v_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch) return;
    
    // Thread-local state
    float curr_x[256];  // Max dim = 256
    float curr_v[256];
    float v_half[256];
    float gamma[256];
    float mu1[256];
    float mu2[256];
    
    // Initialize state
    for (int i = 0; i < dim; ++i) {
        curr_x[i] = x[tid * dim + i];
        curr_v[i] = v[tid * dim + i];
    }
    
    // Integrate sequence
    for (int t = 0; t < seq_len; ++t) {
        const float* f_ptr = &f[tid * seq_len * dim + t * dim];
        float h = dt * 0.5f;  // Half time step
        float effective_dt = dt;
        
        // 0. PARITY FIX: Compute Christoffel at current state for Kick 1
        toroidal_christoffel_full(curr_x, curr_v, dim, R, r, gamma);
        
        // 1. KICK 1 (Half step velocity with friction and Christoffel)
        if (W_forget != nullptr && b_forget != nullptr) {
            for (int i = 0; i < dim; ++i) {
                float z = b_forget[i];
                for (int j = 0; j < dim; ++j) {
                    z += W_forget[i * (2 * dim) + j] * sinf(curr_x[j]);
                    z += W_forget[i * (2 * dim) + (dim + j)] * cosf(curr_x[j]);
                }
                float s = 1.0f / (1.0f + expf(-z));
                mu1[i] = s * FRICTION_SCALE<float>;
            }
        } else {
            for (int i = 0; i < dim; ++i) mu1[i] = 0.0f;
        }
        for (int i = 0; i < dim; ++i) {
            float num = curr_v[i] + h * (f_ptr[i] - gamma[i]);
            v_half[i] = num / (1.0f + h * mu1[i] + EPSILON_STANDARD<float>);
        }
        
        // 2. DRIFT (Full step position)
        for (int i = 0; i < dim; ++i) {
            curr_x[i] += effective_dt * v_half[i];
        }
        
        // 3. Apply toroidal boundary conditions
        for (int i = 0; i < dim; ++i) {
            curr_x[i] = atan2f(sinf(curr_x[i]), cosf(curr_x[i]));
        }
        
        // 4. Compute toroidal Christoffel force at new state
        toroidal_christoffel_full(curr_x, v_half, dim, R, r, gamma);
        
        // 5. KICK 2 (Half step velocity with friction)
        if (W_forget != nullptr && b_forget != nullptr) {
            for (int i = 0; i < dim; ++i) {
                float z = b_forget[i];
                for (int j = 0; j < dim; ++j) {
                    z += W_forget[i * (2 * dim) + j] * sinf(curr_x[j]);
                    z += W_forget[i * (2 * dim) + (dim + j)] * cosf(curr_x[j]);
                }
                float s = 1.0f / (1.0f + expf(-z));
                mu2[i] = s * FRICTION_SCALE<float>;
            }
        } else {
            for (int i = 0; i < dim; ++i) mu2[i] = 0.0f;
        }
        for (int i = 0; i < dim; ++i) {
            float num = v_half[i] + h * (f_ptr[i] - gamma[i]);
            curr_v[i] = num / (1.0f + h * mu2[i] + EPSILON_STANDARD<float>);
        }
        
        // 6. Store outputs
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
    const float* x,
    const float* v,
    const float* f,
    const float* W_forget,
    const float* b_forget,
    float R,
    float r,
    float dt,
    int batch,
    int seq_len,
    int dim,
    float* x_out,
    float* v_out,
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
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "v must be float32");
    TORCH_CHECK(f.scalar_type() == at::kFloat, "f must be float32");
    if (W_forget.defined() && W_forget.numel() > 0) TORCH_CHECK(W_forget.scalar_type() == at::kFloat, "W_forget must be float32");
    if (b_forget.defined() && b_forget.numel() > 0) TORCH_CHECK(b_forget.scalar_type() == at::kFloat, "b_forget must be float32");
    TORCH_CHECK(dim <= 256, "toroidal_leapfrog_fused requires dim <= 256");

    auto options = x.options();
    auto x_out = at::empty({batch, seq_len, dim}, options);
    auto v_out = at::empty({batch, seq_len, dim}, options);

    const float* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<float>() : nullptr;
    const float* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<float>() : nullptr;

    launch_toroidal_leapfrog_fused(
        x.data_ptr<float>(),
        v.data_ptr<float>(),
        f.data_ptr<float>(),
        W_forget_ptr,
        b_forget_ptr,
        static_cast<float>(R),
        static_cast<float>(r),
        static_cast<float>(dt),
        static_cast<int>(batch),
        static_cast<int>(seq_len),
        static_cast<int>(dim),
        x_out.data_ptr<float>(),
        v_out.data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return {x_out, v_out};
}
