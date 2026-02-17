#ifndef GFN_CUDA_UNIVERSAL_INTEGRATOR_CUH
#define GFN_CUDA_UNIVERSAL_INTEGRATOR_CUH

#include "../geometry/geometry_library.cuh"
#include "../physics/physics_library.cuh"

namespace gfn {
namespace cuda {

// Integration Methods
enum class IntegrationMethod {
    HEUN,
    LEAPFROG,
    EULER
};

/**
 * @brief Universal sequence integrator kernel.
 * 
 * Each block processes one batch item + one head.
 * Parallelism is exploited within the head dimension (tid < head_dim).
 */
template <typename scalar_t, IntegrationMethod Method>
__global__ void universal_mlayer_kernel(
    scalar_t* __restrict__ x_seq,        // [batch, seq_len, total_dim]
    scalar_t* __restrict__ v_final,      // [batch, total_dim]
    scalar_t* __restrict__ x_final,      // [batch, total_dim]
    const scalar_t* __restrict__ x_init,  // [batch, total_dim]
    const scalar_t* __restrict__ v_init,  // [batch, total_dim]
    const scalar_t* __restrict__ forces,  // [batch, seq_len, total_dim]
    GeometryParams<scalar_t> geo_p,
    PhysicsParams<scalar_t> phys_p,
    int total_dim,
    int seq_len
) {
    int bid = blockIdx.x; // batch index
    int hid = blockIdx.y; // head index
    int tid = threadIdx.x;
    
    int num_heads = gridDim.y;
    int head_dim = blockDim.x;
    int head_offset = hid * head_dim;

    // Head-local pointers
    const scalar_t* x_ptr = x_init + bid * total_dim + head_offset;
    const scalar_t* v_ptr = v_init + bid * total_dim + head_offset;
    
    // Shared Memory Layout: [rank] + [6 * head_dim]
    extern __shared__ char shared_buf[];
    scalar_t* h_shared = (scalar_t*)shared_buf;
    scalar_t* x_shared = h_shared + geo_p.rank;
    scalar_t* v_shared = x_shared + head_dim;
    scalar_t* f_shared = v_shared + head_dim;
    scalar_t* feat_shared = f_shared + head_dim; // For friction [2*head_dim]
    
    // Init state in registers
    scalar_t curr_x = x_ptr[tid];
    scalar_t curr_v = v_ptr[tid];
    scalar_t curr_h = (phys_p.hyst_enabled && phys_p.hysteresis_settings) ? 
                      phys_p.hysteresis_settings[bid * total_dim + head_offset + tid] : 0.0f;

    // --- Per-Head Parameter Offsetting ---
    GeometryParams<scalar_t> head_geo_p = geo_p;
    PhysicsParams<scalar_t> head_phys_p = phys_p;
    
    const scalar_t dt_eff = phys_p.dt * phys_p.dt_scales[hid];
    const scalar_t local_holo_z = (geo_p.holo_z_ptr) ? geo_p.holo_z_ptr[bid * num_heads + hid] : 0.0f;

    // Geometry offsets
    head_geo_p.U = geo_p.U + hid * head_dim * geo_p.rank;
    head_geo_p.W = geo_p.W + hid * head_dim * geo_p.rank;
    if (geo_p.holo_grad_z) {
        head_geo_p.holo_grad_z = geo_p.holo_grad_z + (bid * num_heads * head_dim + hid * head_dim);
    }

    // Physics offsets
    int feat_dim = (geo_p.topology == Topology::TORUS) ? 2 * head_dim : head_dim;
    
    // Singularity vector offset (matches feature dim)
    if (geo_p.V_w) {
        head_geo_p.V_w = geo_p.V_w + hid * feat_dim;
    }

    head_phys_p.W_forget = phys_p.W_forget + hid * head_dim * feat_dim;
    head_phys_p.b_forget = phys_p.b_forget + hid * head_dim;
    if (phys_p.W_input) {
        head_phys_p.W_input = phys_p.W_input + hid * head_dim * head_dim;
    }
    
    // Hysteresis offsets
    if (phys_p.hyst_enabled) {
        int x_feat_dim = (geo_p.topology == Topology::TORUS) ? 2 * head_dim : head_dim;
        int hyst_in = x_feat_dim + head_dim;
        head_phys_p.hyst_up_w = phys_p.hyst_up_w + hid * head_dim * hyst_in;
        head_phys_p.hyst_up_b = phys_p.hyst_up_b + hid * head_dim;
        head_phys_p.hyst_rd_w = phys_p.hyst_rd_w + hid * head_dim * head_dim;
        head_phys_p.hyst_rd_b = phys_p.hyst_rd_b + hid * head_dim;
    }

    // --- Sequence Loop ---
    for (int t = 0; t < seq_len; ++t) {
        scalar_t f_ext = forces[(bid * seq_len + t) * total_dim + head_offset + tid];
        
        // Sync registers to shared for block-parallel calculations
        x_shared[tid] = curr_x;
        v_shared[tid] = curr_v;
        __syncthreads();

        MLayerState<scalar_t> state = {x_shared, v_shared, &curr_h, f_ext};
        scalar_t gamma, mu, ghost_f;
        
        if (Method == IntegrationMethod::LEAPFROG) {
            // Stage 1: Half-Kick (Implicit Friction + Curvature)
            scalar_t v_sq = (*state.v) * (*state.v);
            scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));
            
            compute_christoffel_distributed(head_geo_p, state, head_dim, local_holo_z, &gamma, h_shared, feat_shared);
            compute_friction_distributed(head_phys_p, state, head_dim, head_geo_p.topology, v_norm, &mu, feat_shared);
            apply_hysteresis_distributed(head_phys_p, state, head_dim, head_geo_p.topology, &ghost_f, feat_shared);
            
            // v_half = (v + h*(F + Fg - G)) / (1 + h*mu)
            scalar_t h = static_cast<scalar_t>(0.5) * dt_eff;
            scalar_t v_half =
                (curr_v + h * (f_ext + ghost_f - gamma)) /
                (static_cast<scalar_t>(1) + h * mu + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
            
            // Stage 2: Drift
            curr_x = apply_boundary_device(curr_x + dt_eff * v_half, head_geo_p.topology);
            x_shared[tid] = curr_x; // Update shared for next kick
            __syncthreads();
            
            // Stage 3: Second Half-Kick
            state.v = &v_half; // Use half-velocity for christoffel
            
            v_sq = v_half * v_half;
            v_norm = sqrt(block_reduce_sum_shared(v_sq));
            
            compute_christoffel_distributed(head_geo_p, state, head_dim, local_holo_z, &gamma, h_shared, feat_shared);
            compute_friction_distributed(head_phys_p, state, head_dim, head_geo_p.topology, v_norm, &mu, feat_shared);
            
            curr_v =
                (v_half + h * (f_ext + ghost_f - gamma)) /
                (static_cast<scalar_t>(1) + h * mu + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
        } 
        else if (Method == IntegrationMethod::HEUN) {
            // Stage 1: Evaluate Dynamics at (x, v)
            scalar_t v_sq = (*state.v) * (*state.v);
            scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));

            compute_christoffel_distributed(head_geo_p, state, head_dim, local_holo_z, &gamma, h_shared, feat_shared);
            compute_friction_distributed(head_phys_p, state, head_dim, head_geo_p.topology, v_norm, &mu, feat_shared);
            apply_hysteresis_distributed(head_phys_p, state, head_dim, head_geo_p.topology, &ghost_f, feat_shared);
            
            scalar_t a1 = f_ext + ghost_f - gamma - mu * curr_v;
            scalar_t x_inter = apply_boundary_device(curr_x + dt_eff * curr_v, head_geo_p.topology);
            scalar_t v_inter = curr_v + dt_eff * a1;
            
            // Stage 2: Evaluate at intermediate point
            x_shared[tid] = x_inter;
            v_shared[tid] = v_inter;
            __syncthreads();
            
            v_sq = v_inter * v_inter;
            v_norm = sqrt(block_reduce_sum_shared(v_sq));
            
            compute_christoffel_distributed(head_geo_p, state, head_dim, local_holo_z, &gamma, h_shared, feat_shared);
            compute_friction_distributed(head_phys_p, state, head_dim, head_geo_p.topology, v_norm, &mu, feat_shared);
            
            scalar_t a2 = f_ext + ghost_f - gamma - mu * v_inter;
            
            curr_x = x_inter; // Heun typically uses x_predictor
            curr_v = curr_v + 0.5f * dt_eff * (a1 + a2);
        }
        else if (Method == IntegrationMethod::EULER) {
            scalar_t v_sq = (*state.v) * (*state.v);
            scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));

            compute_christoffel_distributed(head_geo_p, state, head_dim, local_holo_z, &gamma, h_shared, feat_shared);
            compute_friction_distributed(head_phys_p, state, head_dim, head_geo_p.topology, v_norm, &mu, feat_shared);
            apply_hysteresis_distributed(head_phys_p, state, head_dim, head_geo_p.topology, &ghost_f, feat_shared);
            
            // v_{t+1} = v_t + dt * (F - G - mu*v_t)
            // x_{t+1} = x_t + dt * v_t
            scalar_t a = f_ext + ghost_f - gamma - mu * curr_v;
            
            curr_x = apply_boundary_device(curr_x + dt_eff * curr_v, head_geo_p.topology);
            curr_v = curr_v + dt_eff * a;
        }


        // Store to sequence output
        x_seq[(bid * seq_len + t) * total_dim + head_offset + tid] = curr_x;
    }

    // Final state
    x_final[bid * total_dim + head_offset + tid] = curr_x;
    v_final[bid * total_dim + head_offset + tid] = curr_v;
    if (phys_p.hyst_enabled && phys_p.hysteresis_settings) {
        phys_p.hysteresis_settings[bid * total_dim + head_offset + tid] = curr_h;
    }
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_UNIVERSAL_INTEGRATOR_CUH
