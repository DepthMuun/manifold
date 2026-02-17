#include <torch/extension.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "christoffel_impl.cuh"

namespace gfn {
namespace cuda {

template <typename scalar_t>
__global__ void christoffel_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ V_w,
    int batch_size,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t* __restrict__ grad_v,
    scalar_t* __restrict__ grad_U,  // Shared: use atomicAdd
    scalar_t* __restrict__ grad_W,  // Shared: use atomicAdd
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_V_w
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const scalar_t* grad_out_b = grad_out + b * dim;
    const scalar_t* gamma_b = gamma + b * dim;
    const scalar_t* v_b = v + b * dim;
    const scalar_t* x_b = (x != nullptr) ? (x + b * dim) : nullptr;
    
    scalar_t* g_v_b = grad_v + b * dim;
    scalar_t* g_x_b = (grad_x != nullptr) ? (grad_x + b * dim) : nullptr;

    scalar_t h[256], grad_h[256], grad_q[256];
    // AUDIT FIX: Use double for energy accumulation to improve precision
    double h_energy_acc = 0.0;
    for (int i = 0; i < rank; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dim; ++j) sum += static_cast<double>(U[j * rank + i]) * static_cast<double>(v_b[j]);
        h[i] = static_cast<scalar_t>(sum);
        h_energy_acc += sum * sum;
    }
    scalar_t h_energy = static_cast<scalar_t>(h_energy_acc);
    if (rank > 0) {
        h_energy /= static_cast<scalar_t>(rank);
    }
    scalar_t norm = sqrt(h_energy);
    scalar_t S = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
    
    scalar_t M_plas = static_cast<scalar_t>(1);
    scalar_t v_e = static_cast<scalar_t>(0);
    scalar_t tanh_v_e = static_cast<scalar_t>(0);
    if (plasticity != static_cast<scalar_t>(0)) {
        for (int i = 0; i < dim; ++i) v_e += v_b[i] * v_b[i];
        v_e /= static_cast<scalar_t>(dim);
        tanh_v_e = tanh(v_e);
        M_plas = (static_cast<scalar_t>(1) + plasticity * static_cast<scalar_t>(0.1) * tanh_v_e);
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
    }
    for (int i = 0; i < dim; ++i) {
        scalar_t t = gamma_b[i] / static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>);
        scalar_t grad_raw_i = grad_out_b[i] * (static_cast<scalar_t>(1) - t * t);
        for (int j = 0; j < rank; ++j) {
            scalar_t q_base = h[j] * h[j] * S * M;
            atomicAdd(&grad_W[i * rank + j], grad_raw_i * q_base);
            grad_q[j] += W[i * rank + j] * grad_raw_i;
        }
    }
    
    // AUDIT FIX: Use double for sum_grad_q_h_sq calculation
    double sum_grad_q_h_sq_acc = 0.0;
    for (int i = 0; i < rank; ++i) sum_grad_q_h_sq_acc += static_cast<double>(grad_q[i]) * static_cast<double>(h[i]) * static_cast<double>(h[i]);
    
    scalar_t sum_grad_q_h_sq = static_cast<scalar_t>(sum_grad_q_h_sq_acc);
    scalar_t S_sq_M_norm = static_cast<scalar_t>(0);
    if (norm > EPSILON_STANDARD<scalar_t> && rank > 0) {
        // AUDIT FIX: Add division by rank (norm * rank)
        // dS/dh = -S^2 * h / (norm * rank)
        S_sq_M_norm = M * S * S / (norm * static_cast<scalar_t>(rank));
    }
    for (int i = 0; i < rank; ++i) grad_h[i] = grad_q[i] * h[i] * static_cast<scalar_t>(2) * S * M - sum_grad_q_h_sq * S_sq_M_norm * h[i];
    
    scalar_t dL_dM_plas = sum_grad_q_h_sq * S * M_sing;
    scalar_t plas_scale = plasticity * static_cast<scalar_t>(0.1);
    scalar_t dM_plas_dv_scale = plas_scale * (static_cast<scalar_t>(1) - tanh_v_e * tanh_v_e) * static_cast<scalar_t>(2) / static_cast<scalar_t>(dim);
    for (int i = 0; i < dim; ++i) {
        g_v_b[i] = 0;
        for (int j = 0; j < rank; ++j) {
            atomicAdd(&grad_U[i * rank + j], v_b[i] * grad_h[j]);
            g_v_b[i] += U[i * rank + j] * grad_h[j];
        }
        if (plasticity != static_cast<scalar_t>(0)) {
            g_v_b[i] += dL_dM_plas * dM_plas_dv_scale * v_b[i];
        }
    }

    // Common factor term for singularity gate chain-rule, only meaningful when V_w present
    scalar_t dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
    scalar_t factor = dL_dM_sing * (sing_strength - static_cast<scalar_t>(1)) 
                    * static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) 
                    * soft_m * (static_cast<scalar_t>(1) - soft_m) 
                    * gate * (static_cast<scalar_t>(1) - gate);

    if (V_w != nullptr && grad_V_w != nullptr) {
        // Gradient w.r.t V_w
        for (int i = 0; i < dim; ++i) {
            scalar_t feature = (topology == Topology::TORUS) 
                               ? (x_b ? sin(x_b[i]) : static_cast<scalar_t>(0))
                               : (x_b ? x_b[i] : static_cast<scalar_t>(0));
            atomicAdd(&grad_V_w[i], factor * feature);
        }
    }
    if (g_x_b != nullptr && V_w != nullptr) {
        for (int i = 0; i < dim; ++i) {
            g_x_b[i] = factor * ((topology == Topology::TORUS) ? cos(x_b[i]) * V_w[i] : V_w[i]);
        }
    }
}

} // namespace cuda
} // namespace gfn


std::vector<torch::Tensor> christoffel_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor gamma,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor W,
    torch::Tensor x,
    torch::Tensor V_w,
    double plasticity,
    double sing_thresh,
    double sing_strength,
    int topology,
    double R,
    double r
) {
    int batch_size = v.size(0);
    int dim = v.size(1);
    int rank = U.size(1);

    auto options = v.options();
    auto grad_v = torch::zeros_like(v);
    auto grad_U = torch::zeros_like(U);
    auto grad_W = torch::zeros_like(W);
    auto grad_x = torch::zeros_like(x);
    auto grad_V_w = torch::zeros_like(V_w);

    int threads = 128;
    int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "christoffel_backward_cuda", ([&] {
        gfn::cuda::christoffel_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            x.numel() > 0 ? x.data_ptr<scalar_t>() : nullptr,
            V_w.numel() > 0 ? V_w.data_ptr<scalar_t>() : nullptr,
            batch_size, dim, rank,
            static_cast<scalar_t>(plasticity), static_cast<scalar_t>(sing_thresh), static_cast<scalar_t>(sing_strength),
            static_cast<gfn::cuda::Topology>(topology), static_cast<scalar_t>(R), static_cast<scalar_t>(r),
            grad_v.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            grad_W.data_ptr<scalar_t>(),
            grad_x.numel() > 0 ? grad_x.data_ptr<scalar_t>() : nullptr,
            grad_V_w.numel() > 0 ? grad_V_w.data_ptr<scalar_t>() : nullptr
        );
    }));

    return {grad_v, grad_U, grad_W, grad_x, grad_V_w};
}
