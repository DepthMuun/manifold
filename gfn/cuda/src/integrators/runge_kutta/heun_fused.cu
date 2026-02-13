/**
 * Heun (RK2) Fused Integrator Kernel
 * ====================================
 *
 * Block-parallel Heun integrator for manifold dynamics.
 * 1 block = 1 batch item, blockDim.x = dim.
 *
 * FIX (2026-02-11): Rewritten from single-thread-per-batch to block-parallel.
 *   BUG-1: Old kernel used 1 thread per batch with scalar_t[64] stacks,
 *          limiting dim<=64 and running without dimension parallelism.
 *   BUG-2: Host wrapper used hardcoded scalar_t instead of AT_DISPATCH.
 *   BUG-5: Added W_input parameter for full friction gate parity with Python.
 *
 * Integration scheme (Heun / RK2 Predictor-Corrector):
 *   1. Compute acceleration: a1 = F - Γ(v, x) - μ(x, F) * v
 *   2. Euler predictor: x_pred = x + dt*v, v_pred = v + dt*a1
 *   3. Compute acceleration at predicted state: a2 = F - Γ(v_pred, x_pred) - μ(x_pred, F)*v_pred
 *   4. Corrector (average): x_new = x + dt/2*(v + v_pred), v_new = v + dt/2*(a1 + a2)
 */

#include "../../geometry/christoffel_impl.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gfn {
namespace cuda {

// ============================================================================
// Block Reduction Helpers (same as leapfrog)
// ============================================================================

template <typename scalar_t>
__device__ inline scalar_t warp_reduce_sum_heun(scalar_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t>
__device__ inline scalar_t block_reduce_sum_heun(scalar_t val) {
    static __shared__ scalar_t shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum_heun(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_sum_heun(val);
    
    return val;
}

// ============================================================================
// Distributed Christoffel (block-parallel, matching leapfrog pattern)
// ============================================================================

template <typename scalar_t>
__device__ void christoffel_distributed_heun(
    scalar_t v_val,
    const scalar_t* U,
    const scalar_t* W,
    const scalar_t* x_shared,
    scalar_t* v_shared,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* gamma_val,
    scalar_t* h_shared
) {
    int tid = threadIdx.x;
    if (tid >= dim) return;

    // 1. Torus special case
    if (topology == Topology::TORUS && x_shared != nullptr) {
        *gamma_val = 0.0f;
        // Use shared memory for neighbor access (safe across warps, fixes BUG-6 pattern)
        v_shared[tid] = v_val;
        __syncthreads();
        
        if (tid % 2 == 0 && tid < dim - 1) {
            scalar_t th = x_shared[tid];
            scalar_t v_ph = v_shared[tid + 1];
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
            scalar_t v_th = v_shared[tid - 1];
            scalar_t s, c;
            sincos_scalar(th, &s, &c);
            
            scalar_t denom = R + r * c;
            denom = (denom < CLAMP_MIN_STRONG) ? CLAMP_MIN_STRONG : denom;
            scalar_t term_ph = -(r * s) / (denom + EPSILON_SMOOTH);
            scalar_t g1 = 2.0f * term_ph * v_ph * v_th;
            
            *gamma_val = soft_clamp(g1 * TOROIDAL_CURVATURE_SCALE, CURVATURE_CLAMP);
        }
        return;
    }

    // 2. Low-rank Logic: h_r = sum(U[d, r] * v[d])
    for (int k = 0; k < rank; ++k) {
        scalar_t u_val = U[tid * rank + k];
        scalar_t prod = u_val * v_val;
        
        scalar_t sum = block_reduce_sum_heun(prod);
        
        if (tid == 0) {
            h_shared[k] = sum;
        }
        __syncthreads(); 
    }
    
    // Energy and scaling (thread 0)
    __shared__ scalar_t S_shared;
    __shared__ scalar_t M_shared;
    
    if (tid == 0) {
        scalar_t energy = 0.0f;
        for (int k = 0; k < rank; ++k) energy += h_shared[k] * h_shared[k];
        if (rank > 0) energy /= static_cast<scalar_t>(rank);
        
        scalar_t norm = sqrt(energy);
        S_shared = 1.0f / (1.0f + norm + EPSILON_STANDARD);
        M_shared = 1.0f;
    }
    __syncthreads();
    
    // Compute Gamma: gamma[tid] = sum(W[tid, k] * h_sq[k])
    scalar_t sum_gamma = 0.0f;
    for (int k = 0; k < rank; ++k) {
        scalar_t h_val = h_shared[k];
        scalar_t h_sq = h_val * h_val * S_shared * M_shared;
        sum_gamma += W[tid * rank + k] * h_sq;
    }
    
    *gamma_val = soft_clamp(sum_gamma, CURVATURE_CLAMP);
}

// ============================================================================
// Distributed Friction (block-parallel)
// ============================================================================

template <typename scalar_t>
__device__ void friction_distributed_heun(
    scalar_t x_val,
    scalar_t f_val,
    const scalar_t* W_forget,
    const scalar_t* b_forget,
    const scalar_t* W_input,
    scalar_t* friction_val,
    int dim,
    Topology topology,
    scalar_t velocity_friction_scale,
    scalar_t v_norm,
    scalar_t* features_shared
) {
    int tid = threadIdx.x;
    if (tid >= dim) return;

    // 1. Compute features
    if (topology == Topology::TORUS) {
        scalar_t s, c;
        sincos_scalar(x_val, &s, &c);
        features_shared[tid] = s;
        features_shared[dim + tid] = c;
    } else {
        features_shared[tid] = x_val;
    }
    __syncthreads();
    
    int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    
    // 2. Forget gate: gate = b + W_forget * features
    scalar_t gate_sum = b_forget[tid];
    for (int j = 0; j < feat_dim; ++j) {
        gate_sum += W_forget[tid * feat_dim + j] * features_shared[j];
    }
    
    // 3. Input gate (BUG-5 fix): if W_input present, add input_gate(force)
    if (W_input != nullptr) {
        // Store force in shared for broadcast
        // Reuse features_shared temporarily (after __syncthreads)
        __syncthreads();
        features_shared[tid] = f_val;
        __syncthreads();
        
        scalar_t input_sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            input_sum += W_input[tid * dim + j] * features_shared[j];
        }
        gate_sum += input_sum;
    }
    
    scalar_t base_friction = sigmoid(gate_sum) * FRICTION_SCALE;
    
    // 4. Velocity Scaling
    if (velocity_friction_scale > 0.0f) {
        scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + EPSILON_SMOOTH);
        *friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
    } else {
        *friction_val = base_friction;
    }
}

// ============================================================================
// Heun Integrator Kernel (Block-Parallel: 1 Block per Batch Item)
// ============================================================================

template <typename scalar_t>
__global__ void heun_fused_kernel(
    const scalar_t* __restrict__ x_in,
    const scalar_t* __restrict__ v_in,
    const scalar_t* __restrict__ force,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ W_forget,
    const scalar_t* __restrict__ b_forget,
    const scalar_t* __restrict__ W_input,
    scalar_t* __restrict__ x_out,
    scalar_t* __restrict__ v_out,
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
    scalar_t velocity_friction_scale,
    
    // Hysteresis parameters (Added for parity)
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

    // Shared Memory:
    //   h_shared:        [rank]
    //   features_shared: [2*dim]  (for friction)
    //   x_shared:        [dim]    (for torus/friction x broadcast)
    //   v_shared:        [dim]    (for torus neighbor access)
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
    
    // Integration Loop
    for (int step = 0; step < steps; ++step) {
        // --- Stage 1: Evaluate at current state ---
        
        // Update shared x
        x_shared[tid] = curr_x;
        __syncthreads();
        
        // 0. Ghost Force (Hysteresis)
        scalar_t f_ghost = 0.0f;
        if (hyst_enabled && hyst_readout_w != nullptr) {
            scalar_t* hyst_shared_buf = features_shared; // Reuse features buf
            hyst_shared_buf[tid] = hyst_val;
            __syncthreads();
            
            scalar_t sum = (hyst_readout_b) ? hyst_readout_b[tid] : 0.0f;
            for (int j = 0; j < dim; ++j) {
                sum += hyst_shared_buf[j] * hyst_readout_w[tid * dim + j];
            }
            f_ghost = sum;
            __syncthreads();
        }
        
        // Friction at current position
        scalar_t friction = 0.0f;
        if (W_forget != nullptr) {
            // Calculate v_norm
            scalar_t v_sq = curr_v * curr_v;
            scalar_t v_sum = block_reduce_sum_heun(v_sq);
            scalar_t v_norm = sqrt(v_sum);
            
            friction_distributed_heun(
                curr_x, f_ext, W_forget, b_forget, W_input,
                &friction, dim, topology, velocity_friction_scale, v_norm, features_shared
            );
        }
        __syncthreads();
        
        // Christoffel at current state
        scalar_t gamma1 = 0.0f;
        christoffel_distributed_heun(
            curr_v, U, W, x_shared, v_shared, dim, rank, plasticity,
            sing_thresh, sing_strength, topology, R, r,
            &gamma1, h_shared
        );
        __syncthreads();
        
        // Acceleration 1: a1 = F + F_ghost - Γ - μ*v
        scalar_t acc1 = f_ext + f_ghost - gamma1 - friction * curr_v;
        
        // --- Euler Predictor ---
        scalar_t v_pred = curr_v + effective_dt * acc1;
        scalar_t x_pred = curr_x + effective_dt * curr_v;
        x_pred = apply_boundary_device(x_pred, topology);
        
        // --- Stage 2: Evaluate at predicted state ---
        
        // Update shared x with predicted position
        x_shared[tid] = x_pred;
        __syncthreads();
        
        // Ghost Force at predicted state (Optional: assumes hyst state doesn't change fast within step)
        // We use the same f_ghost for simplicity and stability within the step, 
        // or we could recompute if hyst_val was dynamic (it's not updated until end of step).
        
        // Friction at predicted position
        scalar_t friction2 = friction;
        if (W_forget != nullptr) {
            scalar_t v_sq = v_pred * v_pred;
            scalar_t v_sum = block_reduce_sum_heun(v_sq);
            scalar_t v_norm = sqrt(v_sum);

            friction_distributed_heun(
                x_pred, f_ext, W_forget, b_forget, W_input,
                &friction2, dim, topology, velocity_friction_scale, v_norm, features_shared
            );
        }
        __syncthreads();
        
        // Christoffel at predicted state
        scalar_t gamma2 = 0.0f;
        christoffel_distributed_heun(
            v_pred, U, W, x_shared, v_shared, dim, rank, plasticity,
            sing_thresh, sing_strength, topology, R, r,
            &gamma2, h_shared
        );
        __syncthreads();
        
        // Acceleration 2: a2 = F + F_ghost - Γ_pred - μ_pred * v_pred
        scalar_t acc2 = f_ext + f_ghost - gamma2 - friction2 * v_pred;
        
        // --- Corrector (average of two slopes) ---
        curr_x += (effective_dt / 2.0f) * (curr_v + v_pred);
        curr_v += (effective_dt / 2.0f) * (acc1 + acc2);
        
        // Apply boundary conditions
        curr_x = apply_boundary_device(curr_x, topology);
        
        // Update Hysteresis State (at end of step)
        if (hyst_enabled && hyst_update_w != nullptr) {
            scalar_t* input_shared = features_shared; // Reuse
            if (topology == Topology::TORUS) {
                scalar_t s, c;
                sincos_scalar(curr_x, &s, &c);
                input_shared[tid] = s;
                input_shared[dim + tid] = c;
            } else {
                input_shared[tid] = curr_x;
            }
            __syncthreads();
            
            int offset = (topology == Topology::TORUS) ? 2*dim : dim;
            input_shared[offset + tid] = curr_v;
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

} // namespace cuda
} // namespace gfn

// ============================================================================
// PyTorch C++ Interface
// ============================================================================

using namespace gfn::cuda;

std::vector<torch::Tensor> heun_fused(
    torch::Tensor x,
    torch::Tensor v,
    torch::Tensor force,
    torch::Tensor U,
    torch::Tensor W,
    float dt,
    float dt_scale,
    int steps,
    int topology,
    torch::Tensor W_forget,
    torch::Tensor b_forget,
    torch::Tensor W_input,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    float R,
    float r,
    float velocity_friction_scale,
    torch::Tensor hysteresis_state,
    torch::Tensor hyst_update_w,
    torch::Tensor hyst_update_b,
    torch::Tensor hyst_readout_w,
    torch::Tensor hyst_readout_b,
    float hyst_decay,
    bool hyst_enabled
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(force.is_cuda(), "force must be a CUDA tensor");
    
    int batch_size = x.size(0);
    int dim = x.size(1);
    int rank = U.size(1);
    
    auto x_out = torch::empty_like(x);
    auto v_out = torch::empty_like(v);
    
    // Launch configuration: 1 block per batch item
    dim3 blocks(batch_size);
    dim3 threads(dim);
    
    // Shared memory: h[rank] + features[2*dim] + x[dim] + v[dim]
    size_t shared_mem_size = (rank + 4 * dim) * x.element_size();
    
    // FIX BUG-2: Use AT_DISPATCH for proper type dispatch
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "heun_fused_cuda", ([&] {
        // Prepare optional pointers
        const scalar_t* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<scalar_t>() : nullptr;
        const scalar_t* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<scalar_t>() : nullptr;
        const scalar_t* W_input_ptr = (W_input.numel() > 0) ? W_input.data_ptr<scalar_t>() : nullptr;
        
        scalar_t* hyst_state_ptr = nullptr;
        const scalar_t* h_up_w_ptr = nullptr;
        const scalar_t* h_up_b_ptr = nullptr;
        const scalar_t* h_rd_w_ptr = nullptr;
        const scalar_t* h_rd_b_ptr = nullptr;
        int hyst_in_dim = 0;
        
        if (hyst_enabled) {
             if (hysteresis_state.defined()) hyst_state_ptr = hysteresis_state.data_ptr<scalar_t>();
             if (hyst_update_w.defined()) {
                 h_up_w_ptr = hyst_update_w.data_ptr<scalar_t>();
                 hyst_in_dim = hyst_update_w.size(1);
             }
             if (hyst_update_b.defined()) h_up_b_ptr = hyst_update_b.data_ptr<scalar_t>();
             if (hyst_readout_w.defined()) h_rd_w_ptr = hyst_readout_w.data_ptr<scalar_t>();
             if (hyst_readout_b.defined()) h_rd_b_ptr = hyst_readout_b.data_ptr<scalar_t>();
        }

        heun_fused_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            force.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            W_forget_ptr,
            b_forget_ptr,
            W_input_ptr,
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
            hyst_state_ptr,
            h_up_w_ptr,
            h_up_b_ptr,
            h_rd_w_ptr,
            h_rd_b_ptr,
            static_cast<scalar_t>(hyst_decay),
            hyst_enabled,
            hyst_in_dim
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return {x_out, v_out};
}
