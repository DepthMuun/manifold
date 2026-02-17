#include <torch/extension.h>
#include <cuda_runtime.h>
#include "christoffel_impl.cuh"

namespace gfn {
namespace cuda {

/**
 * Memory-efficient backward pass using atomicAdd for shared parameters.
 * Eliminates [batch, dim, rank] temporary tensors.
 */
template <typename scalar_t>
__global__ void lowrank_christoffel_friction_backward_kernel(
    const scalar_t* __restrict__ grad_out,      // [batch, dim]
    const scalar_t* __restrict__ output,        // [batch, dim]
    const scalar_t* __restrict__ v,             // [batch, dim]
    const scalar_t* __restrict__ U,             // [dim, rank]
    const scalar_t* __restrict__ W,             // [dim, rank]
    const scalar_t* __restrict__ x,             // [batch, dim]
    const scalar_t* __restrict__ V_w,           // [dim] or nullptr
    const scalar_t* __restrict__ force,         // [batch, dim] or nullptr
    const scalar_t* __restrict__ W_forget,      // [dim, feat_dim]
    const scalar_t* __restrict__ b_forget,      // [dim]
    const scalar_t* __restrict__ W_input,       // [dim, dim] or nullptr
    int batch_size,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t velocity_friction_scale, // Added
    // Outputs (atomicAdd for parameters)
    scalar_t* __restrict__ grad_v,              // [batch, dim]
    scalar_t* __restrict__ grad_U,              // [dim, rank]
    scalar_t* __restrict__ grad_W,              // [dim, rank]
    scalar_t* __restrict__ grad_x,              // [batch, dim]
    scalar_t* __restrict__ grad_Vw,             // [dim] or nullptr
    scalar_t* __restrict__ grad_force,          // [batch, dim] or nullptr
    scalar_t* __restrict__ grad_Wf,             // [dim, feat_dim]
    scalar_t* __restrict__ grad_bf,             // [dim]
    scalar_t* __restrict__ grad_Wi              // [dim, dim] or nullptr
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const scalar_t* grad_out_b = grad_out + b * dim;
    const scalar_t* v_b = v + b * dim;
    const scalar_t* x_b = x + b * dim;
    const scalar_t* force_b = (force != nullptr) ? (force + b * dim) : nullptr;
    
    scalar_t gamma_b[64];
    christoffel_device(v_b, U, W, x_b, V_w, dim, rank, plasticity, sing_thresh, sing_strength, topology, R, r, gamma_b);

    // Soft Tanh correction
    scalar_t grad_pre[64]; // Max dim 64 recommended for this kernel
    for(int i = 0; i < dim && i < 64; ++i) {
        scalar_t t = gamma_b[i] / static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>);
        grad_pre[i] = grad_out_b[i] * (static_cast<scalar_t>(1) - t * t);
    }

    // Local gradients for non-shared inputs
    scalar_t* g_v_b = grad_v + b * dim;
    scalar_t* g_x_b = grad_x + b * dim;
    scalar_t* g_f_b = (grad_force != nullptr) ? (grad_force + b * dim) : nullptr;
    
    for(int i = 0; i < dim; ++i) { g_v_b[i] = static_cast<scalar_t>(0); g_x_b[i] = static_cast<scalar_t>(0); if(g_f_b) g_f_b[i] = static_cast<scalar_t>(0); }

    // 1. Friction contribution dL/dmu_i = grad_out[i] * v_b[i]
    scalar_t grad_mu[64];
    for(int i = 0; i < dim && i < 64; ++i) {
        grad_mu[i] = grad_out_b[i] * v_b[i];
    }
    
    // BACKWARD FRICTION (Local to parameter gradients handled via atomicAdd)
    int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    scalar_t features[128];
    if (topology == Topology::TORUS) compute_fourier_features<scalar_t>(features, x_b, dim);
    else vector_copy<scalar_t>(features, x_b, dim);
    
    // Calculate v_norm for friction scaling backward
    scalar_t v_norm = static_cast<scalar_t>(0);
    if (velocity_friction_scale > static_cast<scalar_t>(0)) {
        for(int i=0; i<dim; ++i) v_norm += v_b[i] * v_b[i];
        v_norm = sqrt(v_norm);
    }
    
    for (int i = 0; i < dim; ++i) {
        scalar_t z = b_forget[i];
        for (int j = 0; j < feat_dim; ++j) z += W_forget[i * feat_dim + j] * features[j];
        if (force_b != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) z += W_input[i * dim + j] * force_b[j];
        }
        
        scalar_t s = sigmoid<scalar_t>(z);
        // Base friction without velocity scaling
        scalar_t mu_base = s * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
        
        // Adjust gradient for mu based on velocity scaling
        // If velocity scaling is active, mu = mu_base * (1 + scale * |v|)
        // dL/dmu_base = dL/dmu * (1 + scale * |v|)
        scalar_t dL_dmu_base = grad_mu[i];
        if (velocity_friction_scale > static_cast<scalar_t>(0)) {
             scalar_t scale_factor = static_cast<scalar_t>(1) + velocity_friction_scale * v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
             dL_dmu_base *= scale_factor;
        }
        
        scalar_t dz = dL_dmu_base * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>) * s * (static_cast<scalar_t>(1) - s);
        
        atomicAdd(&grad_bf[i], dz);
        for (int j = 0; j < feat_dim; ++j) atomicAdd(&grad_Wf[i * feat_dim + j], dz * features[j]);
        if (force_b != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) {
                atomicAdd(&grad_Wi[i * dim + j], dz * force_b[j]);
                if (g_f_b) g_f_b[j] += dz * W_input[i * dim + j];
            }
        }
        
        if (topology == Topology::TORUS) {
            for (int j = 0; j < dim; ++j) {
                scalar_t d_sin = W_forget[i * feat_dim + j] * dz;
                scalar_t d_cos = W_forget[i * feat_dim + (dim + j)] * dz;
                g_x_b[j] += d_sin * cos(x_b[j]) - d_cos * sin(x_b[j]);
            }
        } else {
            for (int j = 0; j < dim; ++j) g_x_b[j] += W_forget[i * feat_dim + j] * dz;
        }
    }
    
    // BACKWARD CHRISTOFFEL
    // Temporary locals for U, W gradients to use atomicAdd
    scalar_t h[64], grad_h[64], grad_q[64];
    scalar_t h_energy = static_cast<scalar_t>(0);
    for (int i = 0; i < rank; ++i) {
        scalar_t sum = static_cast<scalar_t>(0);
        for (int j = 0; j < dim; ++j) sum += U[j * rank + i] * v_b[j];
        h[i] = sum;
        h_energy += sum * sum;
    }
    if (rank > 0) {
        h_energy /= static_cast<scalar_t>(rank);
    }
    scalar_t norm_h = sqrt(h_energy);
    scalar_t S = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm_h + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
    
    scalar_t M_plas = static_cast<scalar_t>(1);
    if (plasticity != static_cast<scalar_t>(0)) {
        scalar_t v_e = static_cast<scalar_t>(0);
        for (int i = 0; i < dim; ++i) v_e += v_b[i] * v_b[i];
        v_e /= static_cast<scalar_t>(dim);
        M_plas = (static_cast<scalar_t>(1) + plasticity * static_cast<scalar_t>(0.1) * tanh(v_e));
    }
    
    scalar_t M_sing = static_cast<scalar_t>(1);
    scalar_t pot = static_cast<scalar_t>(0), gate = static_cast<scalar_t>(0), soft_m = static_cast<scalar_t>(0);
    if (x_b != nullptr && V_w != nullptr) {
        if (topology == Topology::TORUS) { for (int i = 0; i < dim; ++i) pot += sin(x_b[i]) * V_w[i]; }
        else { for (int i = 0; i < dim; ++i) pot += x_b[i] * V_w[i]; }
        gate = sigmoid<scalar_t>(pot);
        soft_m = sigmoid<scalar_t>(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));
        M_sing = (static_cast<scalar_t>(1) + (sing_strength - static_cast<scalar_t>(1)) * soft_m);
    }
    scalar_t M = M_plas * M_sing;

    for (int j = 0; j < rank; ++j) {
        grad_q[j] = static_cast<scalar_t>(0);
        scalar_t q_base = h[j] * h[j] * S * M;
        for (int i = 0; i < dim; ++i) {
            atomicAdd(&grad_W[i * rank + j], grad_pre[i] * q_base);
            grad_q[j] += W[i * rank + j] * grad_pre[i];
        }
    }
    
    scalar_t sum_grad_q_h_sq = static_cast<scalar_t>(0);
    for (int i = 0; i < rank; ++i) sum_grad_q_h_sq += grad_q[i] * h[i] * h[i];
    scalar_t S_sq_M_norm = (norm_h > EPSILON_STANDARD<scalar_t>) ? (M * S * S / norm_h) : static_cast<scalar_t>(0);
    for (int i = 0; i < rank; ++i) grad_h[i] = grad_q[i] * h[i] * static_cast<scalar_t>(2) * S * M - sum_grad_q_h_sq * S_sq_M_norm * h[i];
    
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < rank; ++j) {
            atomicAdd(&grad_U[i * rank + j], v_b[i] * grad_h[j]);
            g_v_b[i] += U[i * rank + j] * grad_h[j];
        }
    }

    // Mu contribution to grad_v
    scalar_t mu_b[64];
    compute_friction<scalar_t>(x_b, force_b, W_forget, b_forget, W_input, dim, topology, velocity_friction_scale, v_norm, mu_b);
    for(int i = 0; i < dim; ++i) g_v_b[i] += mu_b[i] * grad_out_b[i];
    
    // Add gradient from velocity scaling of friction: d(mu)/dv
    if (velocity_friction_scale > static_cast<scalar_t>(0) && v_norm > EPSILON_STANDARD<scalar_t>) {
        scalar_t v_scale_grad_factor = velocity_friction_scale / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
        for(int i=0; i<dim; ++i) {
            // mu[i] = mu_base[i] * (1 + scale * |v|)
            // d(mu[i])/dv[j] = mu_base[i] * scale * v[j]/|v|
            // contribution to g_v_b[j] is sum_i (grad_out_b[i] * v_b[i]) * d(mu[i])/dv[j] ... wait
            // The term is F_fric = mu * v
            // d(F_fric[i])/dv[j] (for i!=j) = v[i] * d(mu[i])/dv[j]
            // d(F_fric[i])/dv[i] = mu[i] + v[i] * d(mu[i])/dv[i]
            // We already added mu[i] * grad_out_b[i] (this is the first term)
            // Now we need sum_k (grad_out_b[k] * v_b[k]) * d(mu[k])/dv[j]
            // d(mu[k])/dv[j] = mu_base[k] * scale * v[j] / |v|
            // So we need sum_k (grad_out_b[k] * v_b[k] * mu_base[k]) * (scale * v[j] / |v|)
            
            // Recompute mu_base[i] (we didn't store it)
            // Or reuse mu_b[i] which is mu_base * (1+...)
            // mu_base[i] = mu_b[i] / (1 + scale * |v|)
            
            scalar_t scale_term = 1.0f + v_scale_grad_factor * v_norm;
            scalar_t mu_base_i = mu_b[i] / scale_term;
            
            // The sum is over k. Wait, this loop is over i.
            // I need to accumulate the common factor first.
            // common_factor = sum_k (grad_out_b[k] * v_b[k] * mu_base[k])
            // This needs a separate loop.
        }
        
        scalar_t common_sum = static_cast<scalar_t>(0);
        scalar_t scale_term = static_cast<scalar_t>(1) + v_scale_grad_factor * v_norm;
        for(int k=0; k<dim; ++k) {
            scalar_t mu_base_k = mu_b[k] / scale_term;
            common_sum += grad_out_b[k] * v_b[k] * mu_base_k;
        }
        
        scalar_t factor = common_sum * v_scale_grad_factor / v_norm;
        for(int j=0; j<dim; ++j) {
            g_v_b[j] += factor * v_b[j];
        }
    }
}

} // namespace cuda
} // namespace gfn

using namespace gfn::cuda;

std::vector<torch::Tensor> lowrank_christoffel_friction_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor output,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor W,
    torch::Tensor x,
    torch::Tensor V_w,
    torch::Tensor force,
    torch::Tensor W_forget,
    torch::Tensor b_forget,
    torch::Tensor W_input,
    double plasticity,
    double sing_thresh,
    double sing_strength,
    int topology,
    double R,
    double r,
    double velocity_friction_scale
) {
    const int batch_size = v.size(0);
    const int dim = v.size(1);
    const int rank = U.size(-1);
    // const int feat_dim = (topology == 1) ? 2 * dim : dim;

    auto grad_v = torch::zeros_like(v);
    auto grad_x = torch::zeros_like(x);
    auto grad_force = (force.numel() > 0) ? torch::zeros_like(force) : torch::empty(0, force.options());
    
    // Parameters use direct accumulation via atomicAdd
    auto grad_U = torch::zeros_like(U);
    auto grad_W = torch::zeros_like(W);
    auto grad_Wf = torch::zeros_like(W_forget);
    auto grad_bf = torch::zeros_like(b_forget);
    auto grad_Wi = (W_input.numel() > 0) ? torch::zeros_like(W_input) : torch::empty(0, v.options());
    auto grad_Vw = (V_w.numel() > 0) ? torch::zeros_like(V_w) : torch::empty(0, v.options());

    int threads = 128; // Reduced threads to increase register availability
    int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "lowrank_christoffel_friction_backward_cuda", ([&] {
        gfn::cuda::lowrank_christoffel_friction_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            V_w.numel() > 0 ? V_w.data_ptr<scalar_t>() : nullptr,
            force.numel() > 0 ? force.data_ptr<scalar_t>() : nullptr,
            W_forget.data_ptr<scalar_t>(),
            b_forget.data_ptr<scalar_t>(),
            W_input.numel() > 0 ? W_input.data_ptr<scalar_t>() : nullptr,
            batch_size, dim, rank,
            static_cast<scalar_t>(plasticity), static_cast<scalar_t>(sing_thresh), static_cast<scalar_t>(sing_strength),
            static_cast<Topology>(topology), static_cast<scalar_t>(R), static_cast<scalar_t>(r),
            static_cast<scalar_t>(velocity_friction_scale),
            grad_v.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            grad_W.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_Vw.numel() > 0 ? grad_Vw.data_ptr<scalar_t>() : nullptr,
            grad_force.numel() > 0 ? grad_force.data_ptr<scalar_t>() : nullptr,
            grad_Wf.data_ptr<scalar_t>(),
            grad_bf.data_ptr<scalar_t>(),
            grad_Wi.numel() > 0 ? grad_Wi.data_ptr<scalar_t>() : nullptr
        );
    }));

    return { grad_v, grad_U, grad_W, grad_x, grad_Vw, grad_force, grad_Wf, grad_bf, grad_Wi };
}
