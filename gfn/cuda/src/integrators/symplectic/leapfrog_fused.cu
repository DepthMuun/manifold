#include "../geometry/christoffel_impl.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace gfn {
namespace cuda {

// ============================================================================
// Block Reduction Helper
// ============================================================================

template <typename scalar_t>
__device__ inline scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t>
__device__ inline scalar_t block_reduce_sum(scalar_t val) {
    static __shared__ scalar_t shared[32]; // Max 32 warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// ============================================================================
// Distributed Helper Functions
// ============================================================================

// Distributed Christoffel: Computes gamma[tid] using distributed v[tid]
template <typename scalar_t>
__device__ void christoffel_distributed(
    scalar_t v_val,      // v[tid]
    const scalar_t* U,   // [dim, rank]
    const scalar_t* W,   // [dim, rank]
    const scalar_t* x_shared, // [dim] in shared memory (for torus check)
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* gamma_val, // Output gamma[tid]
    scalar_t* h_shared,  // Shared memory for h [rank]
    scalar_t* v_shared   // Shared memory for v [dim] (for torus cross-warp access)
) {
    int tid = threadIdx.x;
    if (tid >= dim) return;

    // 1. Torus special case
    if (topology == Topology::TORUS && x_shared != nullptr) {
        *gamma_val = 0.0f;
        // FIX BUG-6: Use shared memory for neighbor access instead of warp shuffles.
        // __shfl_down_sync / __shfl_up_sync only work within a 32-thread warp.
        // When dim > 32, threads in different warps cannot communicate via shuffles.
        // v_shared is dim-sized shared memory passed from the kernel.
        v_shared[tid] = v_val;
        __syncthreads();
        
        if (tid % 2 == 0 && tid < dim - 1) {
            scalar_t th = x_shared[tid];
            scalar_t v_ph = v_shared[tid + 1]; // Safe: reads from shared mem
            scalar_t s, c;
            sincos_scalar(th, &s, &c);
            
            scalar_t denom = R + r * c;
            denom = (denom < CLAMP_MIN_STRONG) ? CLAMP_MIN_STRONG : denom;
            scalar_t term_th = denom * s / (r + EPSILON_SMOOTH);
            scalar_t g0 = term_th * (v_ph * v_ph);
            
            *gamma_val = soft_clamp(g0 * TOROIDAL_CURVATURE_SCALE, CURVATURE_CLAMP);
        } else if (tid % 2 != 0) {
            scalar_t th = x_shared[tid - 1];
            scalar_t v_ph = v_val;
            scalar_t v_th = v_shared[tid - 1]; // Safe: reads from shared mem
            scalar_t s, c;
            sincos_scalar(th, &s, &c);
            
            scalar_t denom = R + r * c;
            denom = (denom < CLAMP_MIN_STRONG) ? CLAMP_MIN_STRONG : denom;
            scalar_t term_ph = -(r * s) / (denom + EPSILON_SMOOTH);
            scalar_t g1 = 2.0f * term_ph * v_ph * v_th;
            
            *gamma_val = soft_clamp(g1 * TOROIDAL_CURVATURE_SCALE, CURVATURE_CLAMP);
        }
        __syncthreads(); // Ensure all threads finish before v_shared is reused
        return;
    }

    // 2. Low-rank Logic
    // h_r = sum(U[d, r] * v[d])
    
    // Iterate over ranks
    for (int k = 0; k < rank; ++k) {
        scalar_t u_val = U[tid * rank + k];
        scalar_t prod = u_val * v_val;
        
        scalar_t sum = block_reduce_sum(prod);
        
        if (tid == 0) {
            h_shared[k] = sum;
        }
        __syncthreads(); 
    }
    
    // Now h_shared is populated.
    // Calculate Energy and Scalars (Thread 0 does it, or all?)
    __shared__ scalar_t S_shared;
    __shared__ scalar_t M_shared;
    
    if (tid == 0) {
        scalar_t energy = 0.0f;
        for (int k = 0; k < rank; ++k) energy += h_shared[k] * h_shared[k];
        if (rank > 0) energy /= static_cast<scalar_t>(rank);
        
        scalar_t norm = sqrt(energy);
        S_shared = 1.0f / (1.0f + norm + EPSILON_STANDARD);
        
        scalar_t M = 1.0f;
        M_shared = M;
    }
    __syncthreads();
    
    // Compute Gamma
    // gamma[tid] = sum(W[tid, k] * h_sq[k])
    scalar_t sum_gamma = 0.0f;
    for (int k = 0; k < rank; ++k) {
        scalar_t h_val = h_shared[k];
        scalar_t h_sq = h_val * h_val * S_shared * M_shared;
        sum_gamma += W[tid * rank + k] * h_sq;
    }
    
    *gamma_val = soft_clamp(sum_gamma, CURVATURE_CLAMP);
}


// Distributed Friction
template <typename scalar_t>
__device__ void friction_distributed(
    scalar_t x_val,
    scalar_t f_val, // force[tid]
    const scalar_t* W_forget, // [dim, feat_dim]
    const scalar_t* b_forget, // [dim]
    const scalar_t* W_input,  // [dim, dim] (Optional)
    scalar_t* friction_val,
    int dim,
    Topology topology,
    scalar_t velocity_friction_scale,
    scalar_t v_norm,
    scalar_t* features_shared // [feat_dim]
) {
    int tid = threadIdx.x;
    if (tid >= dim) return;

    // 1. Compute features and store in shared
    if (topology == Topology::TORUS) {
        // [sin, cos]
        scalar_t s, c;
        sincos_scalar(x_val, &s, &c);
        features_shared[tid] = s;
        features_shared[dim + tid] = c;
    } else {
        features_shared[tid] = x_val;
    }
    __syncthreads();
    
    int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    
    // 2. Compute gate
    // gate[tid] = b[tid] + sum(W[tid, j] * feat[j])
    scalar_t sum = b_forget[tid];
    for (int j = 0; j < feat_dim; ++j) {
        sum += W_forget[tid * feat_dim + j] * features_shared[j];
    }
    
    // 3. Input Gate (FIX: Added support)
    if (W_input != nullptr) {
        // We need force vector in shared memory to do matrix-vector product
        // Reuse features_shared temporarily (safe because we are done with step 2)
        __syncthreads();
        features_shared[tid] = f_val; 
        __syncthreads();
        
        for (int j = 0; j < dim; ++j) {
            sum += W_input[tid * dim + j] * features_shared[j];
        }
    }
    
    scalar_t base_friction = sigmoid(sum) * FRICTION_SCALE;
    
    // 4. Velocity Scaling (FIX: Added support)
    if (velocity_friction_scale > 0.0f) {
        scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + EPSILON_SMOOTH);
        *friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
    } else {
        *friction_val = base_friction;
    }
}


// ============================================================================
// Leapfrog Integrator Kernel (Kick-Drift-Kick with Implicit Friction)
// Block Parallel Version: 1 Block per Batch Item
// ============================================================================

template <typename scalar_t>
__global__ void leapfrog_fused_kernel(
    const scalar_t* __restrict__ x_in,       // [batch, dim]
    const scalar_t* __restrict__ v_in,       // [batch, dim]
    const scalar_t* __restrict__ force,      // [batch, dim]
    const scalar_t* __restrict__ U,          // [dim, rank]
    const scalar_t* __restrict__ W,          // [dim, rank]
    const scalar_t* __restrict__ W_forget,   // [dim, feature_dim]
    const scalar_t* __restrict__ b_forget,   // [dim]
    const scalar_t* __restrict__ W_input,    // [dim, dim] (Added)
    scalar_t* __restrict__ x_out,            // [batch, dim]
    scalar_t* __restrict__ v_out,            // [batch, dim]
    int batch_size,
    int dim,
    int rank,
    scalar_t dt,
    scalar_t dt_scale,
    int steps,
    int topology_id,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    scalar_t R,
    scalar_t r,
    scalar_t velocity_friction_scale, // (Added)
    
    // Hysteresis parameters
    scalar_t* __restrict__ hysteresis_state,     // [batch, dim] (RW)
    const scalar_t* __restrict__ hyst_update_w,  // [dim, in_dim]
    const scalar_t* __restrict__ hyst_update_b,  // [dim]
    const scalar_t* __restrict__ hyst_readout_w, // [dim, dim]
    const scalar_t* __restrict__ hyst_readout_b, // [dim]
    scalar_t hyst_decay,
    bool hyst_enabled,
    int hyst_in_dim
) {
    // 1 Block per Batch Item
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size || tid >= dim) return;

    // Shared Memory Allocation
    // We need:
    // - h [rank]
    // - features [2*dim] (for friction)
    // - x_shared [dim] (for torus/friction)
    // - v_shared [dim] (for torus cross-warp access in christoffel)
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* h_shared = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* features_shared = h_shared + rank;
    scalar_t* x_shared = features_shared + (2 * dim);
    scalar_t* v_shared = x_shared + dim;
    
    // Register State
    scalar_t curr_x = x_in[bid * dim + tid];
    scalar_t curr_v = v_in[bid * dim + tid];
    scalar_t f_ext = force[bid * dim + tid];
    scalar_t hyst_val = 0.0f;
    
    if (hyst_enabled && hysteresis_state != nullptr) {
        hyst_val = hysteresis_state[bid * dim + tid];
    }
    
    Topology topology = static_cast<Topology>(topology_id);
    scalar_t effective_dt = dt * dt_scale;
    scalar_t step_h = 0.5f * effective_dt;
    
    // Integration Loop
    for (int step = 0; step < steps; ++step) {
        // Update shared x for global visibility (needed for torus/friction)
        x_shared[tid] = curr_x;
        __syncthreads();
        
        // 0. Ghost Force
        scalar_t f_ghost = 0.0f;
        if (hyst_enabled && hyst_readout_w != nullptr) {
            scalar_t* hyst_shared_buf = features_shared; 
            hyst_shared_buf[tid] = hyst_val;
            __syncthreads();
            
            scalar_t sum = (hyst_readout_b) ? hyst_readout_b[tid] : 0.0f;
            for (int j = 0; j < dim; ++j) {
                sum += hyst_shared_buf[j] * hyst_readout_w[tid * dim + j];
            }
            f_ghost = sum;
            __syncthreads();
        }
        
        // 1. Friction
        scalar_t friction = 0.0f;
        if (W_forget != nullptr) {
            // Calculate v_norm for velocity scaling
            scalar_t v_sq = curr_v * curr_v;
            scalar_t v_sum = block_reduce_sum(v_sq);
            scalar_t v_norm = sqrt(v_sum);
            
            friction_distributed(
                curr_x, f_ext, W_forget, b_forget, W_input, 
                &friction, dim, topology, velocity_friction_scale, v_norm, features_shared
            );
        }
        __syncthreads(); 
        
        // 2. Christoffel
        scalar_t gamma = 0.0f;
        christoffel_distributed(
            curr_v, U, W, x_shared, dim, rank, plasticity, 
            sing_thresh, sing_strength, topology, R, r, 
            &gamma, h_shared, features_shared
        );
        
        // 3. Kick 1
        scalar_t total_force = f_ext + f_ghost;
        scalar_t num = curr_v + step_h * (total_force - gamma);
        scalar_t den = 1.0f + step_h * friction;
        curr_v = safe_divide(num, den, EPSILON_STANDARD);
        
        // 4. Drift
        curr_x += effective_dt * curr_v;
        curr_x = apply_boundary_device(curr_x, topology);
        
        // Update shared x again
        x_shared[tid] = curr_x;
        __syncthreads();
        
        // 5. Friction at new pos
        if (W_forget != nullptr) {
            // Recalculate v_norm at half-step velocity (curr_v is already updated)
            scalar_t v_sq = curr_v * curr_v;
            scalar_t v_sum = block_reduce_sum(v_sq);
            scalar_t v_norm = sqrt(v_sum);

            friction_distributed(
                curr_x, f_ext, W_forget, b_forget, W_input,
                &friction, dim, topology, velocity_friction_scale, v_norm, features_shared
            );
        }
        __syncthreads();
        
        // 6. Christoffel at new pos
        christoffel_distributed(
            curr_v, U, W, x_shared, dim, rank, plasticity,
            sing_thresh, sing_strength, topology, R, r,
            &gamma, h_shared, features_shared
        );
        
        // 7. Kick 2
        // FIX BUG-4: Recompute ghost force at new state if hysteresis state was updated
        scalar_t total_force2 = f_ext;
        if (hyst_enabled && hyst_readout_w != nullptr) {
            scalar_t* hyst_shared_buf2 = features_shared;
            hyst_shared_buf2[tid] = hyst_val;
            __syncthreads();
            
            scalar_t sum2 = (hyst_readout_b) ? hyst_readout_b[tid] : 0.0f;
            for (int j = 0; j < dim; ++j) {
                sum2 += hyst_shared_buf2[j] * hyst_readout_w[tid * dim + j];
            }
            total_force2 += sum2;
            __syncthreads();
        }
        num = curr_v + step_h * (total_force2 - gamma);
        den = 1.0f + step_h * friction;
        curr_v = safe_divide(num, den, EPSILON_STANDARD);
        
        // 8. Update Hysteresis
        if (hyst_enabled && hyst_update_w != nullptr) {
            scalar_t* input_shared = features_shared; // reuse
            if (topology == Topology::TORUS) {
                scalar_t s, c;
                sincos_scalar(curr_x, &s, &c);
                input_shared[tid] = s;
                input_shared[dim + tid] = c;
            } else {
                input_shared[tid] = curr_x;
            }
            __syncthreads();
            
            if (topology == Topology::TORUS) {
                 input_shared[2*dim + tid] = curr_v;
            } else {
                 input_shared[dim + tid] = curr_v;
            }
            __syncthreads();
            
            scalar_t sum = (hyst_update_b) ? hyst_update_b[tid] : 0.0f;
            for (int j = 0; j < hyst_in_dim; ++j) {
                sum += input_shared[j] * hyst_update_w[tid * hyst_in_dim + j];
            }
            
            hyst_val = hyst_val * hyst_decay + tanhf(sum);
        }
    }
    
    // Store results
    x_out[bid * dim + tid] = curr_x;
    v_out[bid * dim + tid] = curr_v;
    
    if (hyst_enabled && hysteresis_state != nullptr) {
        hysteresis_state[bid * dim + tid] = hyst_val;
    }
}

} 
} 

using namespace gfn::cuda;

// ============================================================================
// Host Wrapper
// ============================================================================

std::vector<torch::Tensor> leapfrog_fused(
    torch::Tensor x, torch::Tensor v, torch::Tensor force,
    torch::Tensor U, torch::Tensor W,
    float dt, float dt_scale, int steps, int topology,
    torch::Tensor W_forget, torch::Tensor b_forget,
    torch::Tensor W_input,
    float plasticity, float sing_thresh, float sing_strength, float R, float r,
    float velocity_friction_scale,
    torch::Tensor hysteresis_state,
    torch::Tensor hyst_update_w,
    torch::Tensor hyst_update_b,
    torch::Tensor hyst_readout_w,
    torch::Tensor hyst_readout_b,
    float hyst_decay,
    bool hyst_enabled
) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.size(0) == v.size(0), "Batch size mismatch");
    TORCH_CHECK(x.size(1) == v.size(1), "Dimension mismatch");

    int batch_size = x.size(0);
    int dim = x.size(1);
    int rank = U.size(1);
    
    auto x_out = torch::empty_like(x);
    auto v_out = torch::empty_like(v);
    
    // Optional pointers
    const void* W_forget_ptr = nullptr;
    const void* b_forget_ptr = nullptr;
    if (W_forget.defined()) W_forget_ptr = W_forget.data_ptr();
    if (b_forget.defined()) b_forget_ptr = b_forget.data_ptr();
    
    const void* W_input_ptr = nullptr;
    if (W_input.defined()) W_input_ptr = W_input.data_ptr();
    
    void* hyst_state_ptr = nullptr;
    const void* hyst_up_w_ptr = nullptr;
    const void* hyst_up_b_ptr = nullptr;
    const void* hyst_read_w_ptr = nullptr;
    const void* hyst_read_b_ptr = nullptr;
    
    if (hyst_enabled) {
        if (hysteresis_state.defined()) hyst_state_ptr = hysteresis_state.data_ptr();
        if (hyst_update_w.defined()) hyst_up_w_ptr = hyst_update_w.data_ptr();
        if (hyst_update_b.defined()) hyst_up_b_ptr = hyst_update_b.data_ptr();
        if (hyst_readout_w.defined()) hyst_read_w_ptr = hyst_readout_w.data_ptr();
        if (hyst_readout_b.defined()) hyst_read_b_ptr = hyst_readout_b.data_ptr();
    }
    
    int hyst_in_dim = 0;
    if (hyst_enabled && hyst_update_w.defined()) {
        hyst_in_dim = hyst_update_w.size(1);
    }

    // Launch configuration
    dim3 blocks(batch_size);
    dim3 threads(dim);
    // Dynamic shared memory calculation:
    // h_shared: rank
    // features_shared: 2*dim
    // x_shared: dim
    // v_shared: dim
    // Total: rank + 4*dim
    size_t shared_mem_size = (rank + 4 * dim) * x.element_size();
    
    // Dispatch
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "leapfrog_fused_cuda", ([&] {
        leapfrog_fused_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            force.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t*>(W_forget_ptr),
            reinterpret_cast<const scalar_t*>(b_forget_ptr),
            reinterpret_cast<const scalar_t*>(W_input_ptr),
            x_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(),
            batch_size,
            dim,
            rank,
            static_cast<scalar_t>(dt),
            static_cast<scalar_t>(dt_scale),
            steps,
            topology,
            static_cast<scalar_t>(plasticity),
            static_cast<scalar_t>(sing_thresh),
            static_cast<scalar_t>(sing_strength),
            static_cast<scalar_t>(R),
            static_cast<scalar_t>(r),
            static_cast<scalar_t>(velocity_friction_scale),
            reinterpret_cast<scalar_t*>(hyst_state_ptr),
            reinterpret_cast<const scalar_t*>(hyst_up_w_ptr),
            reinterpret_cast<const scalar_t*>(hyst_up_b_ptr),
            reinterpret_cast<const scalar_t*>(hyst_read_w_ptr),
            reinterpret_cast<const scalar_t*>(hyst_read_b_ptr),
            static_cast<scalar_t>(hyst_decay),
            hyst_enabled,
            hyst_in_dim
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    return {x_out, v_out};
}
