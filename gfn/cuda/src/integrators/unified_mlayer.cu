#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "universal_integrator.cuh"
#include "../common/params.cuh"

namespace gfn {
namespace cuda {

using namespace gfn::cuda;

// ============================================================================
// Host Wrapper: Unified MLayer Fused
// ============================================================================

std::vector<torch::Tensor> unified_mlayer_fused_cuda(
    torch::Tensor x, torch::Tensor v, torch::Tensor forces,
    torch::Tensor U_stack, torch::Tensor W_stack,
    torch::Tensor W_forget, torch::Tensor b_forget, torch::Tensor W_input,
    torch::Tensor V_w,
    float dt, torch::Tensor dt_scales, int topology,
    float plasticity, float sing_thresh, float sing_strength, float R, float r,
    float velocity_friction_scale,
    float thermo_alpha, float thermo_temp,
    torch::Tensor holo_z, torch::Tensor holo_grad_z,
    torch::Tensor hysteresis_state,
    torch::Tensor hyst_update_w, torch::Tensor hyst_update_b,
    torch::Tensor hyst_readout_w, torch::Tensor hyst_readout_b,
    float hyst_decay, bool hyst_enabled
) {
    int batch_size = x.size(0);
    int total_dim = x.size(1);
    int seq_len = forces.size(1);
    int num_heads = dt_scales.size(0);
    int head_dim = total_dim / num_heads;
    int rank = U_stack.size(2);
    
    auto x_seq = torch::empty({batch_size, seq_len, total_dim}, x.options());
    auto x_final = torch::empty_like(x);
    auto v_final = torch::empty_like(v);
    
    dim3 blocks(batch_size, num_heads);
    dim3 threads(head_dim);
    
    // Shared memory size calculation
    // [rank] + [6 * head_dim]
    size_t shared_mem_size = (rank + 6 * head_dim) * x.element_size();
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "unified_mlayer_fused", ([&] {
        // 1. Pack Geometry Params
        GeometryParams<scalar_t> geo_p;
        geo_p.U = U_stack.data_ptr<scalar_t>();
        geo_p.W = W_stack.data_ptr<scalar_t>();
        geo_p.rank = rank;
        geo_p.topology = static_cast<Topology>(topology);
        geo_p.torus_R = static_cast<scalar_t>(R);
        geo_p.torus_r = static_cast<scalar_t>(r);
        geo_p.plasticity = static_cast<scalar_t>(plasticity);
        geo_p.sing_thresh = static_cast<scalar_t>(sing_thresh);
        geo_p.sing_strength = static_cast<scalar_t>(sing_strength);
        geo_p.thermo_alpha = static_cast<scalar_t>(thermo_alpha);
        geo_p.thermo_temp = static_cast<scalar_t>(thermo_temp);
        geo_p.holo_z_ptr = (holo_z.defined() ? holo_z.data_ptr<scalar_t>() : nullptr);
        geo_p.holo_grad_z = (holo_grad_z.defined() ? holo_grad_z.data_ptr<scalar_t>() : nullptr);
        geo_p.V_w = (V_w.defined() ? V_w.data_ptr<scalar_t>() : nullptr);

        // 2. Pack Physics Params
        PhysicsParams<scalar_t> phys_p;
        phys_p.dt = static_cast<scalar_t>(dt);
        phys_p.dt_scales = dt_scales.data_ptr<scalar_t>();
        phys_p.W_forget = W_forget.data_ptr<scalar_t>();
        phys_p.b_forget = b_forget.data_ptr<scalar_t>();
        phys_p.W_input = (W_input.defined() ? W_input.data_ptr<scalar_t>() : nullptr);
        phys_p.v_fric_scale = static_cast<scalar_t>(velocity_friction_scale);
        phys_p.hyst_enabled = hyst_enabled;
        phys_p.hyst_decay = static_cast<scalar_t>(hyst_decay);
        phys_p.hysteresis_settings = (hyst_enabled ? hysteresis_state.data_ptr<scalar_t>() : nullptr);
        phys_p.hyst_up_w = (hyst_enabled ? hyst_update_w.data_ptr<scalar_t>() : nullptr);
        phys_p.hyst_up_b = (hyst_enabled ? hyst_update_b.data_ptr<scalar_t>() : nullptr);
        phys_p.hyst_rd_w = (hyst_enabled ? hyst_readout_w.data_ptr<scalar_t>() : nullptr);
        phys_p.hyst_rd_b = (hyst_enabled ? hyst_readout_b.data_ptr<scalar_t>() : nullptr);

        // 3. Launch Universal Kernel (Defaulting to Leapfrog for MLayer)
        universal_mlayer_kernel<scalar_t, IntegrationMethod::LEAPFROG><<<blocks, threads, shared_mem_size>>>(
            x_seq.data_ptr<scalar_t>(),
            v_final.data_ptr<scalar_t>(),
            x_final.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            forces.data_ptr<scalar_t>(),
            geo_p,
            phys_p,
            total_dim,
            seq_len
        );
    }));
    
    return {x_final, v_final, x_seq};
}

} // namespace cuda
} // namespace gfn


