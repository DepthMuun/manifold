from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA source files
cuda_src_dir = os.path.join(os.path.dirname(__file__), 'src')

# List all CUDA source files
cuda_sources = [
    'cuda_kernels.cpp',
    # Geometry kernels
    'src/geometry/lowrank_christoffel.cu',
    'src/geometry/lowrank_christoffel_backward.cu',
    # Integrator kernels - Symplectic
    'src/integrators/symplectic/leapfrog_fused.cu',
    'src/integrators/symplectic/leapfrog_backward.cu',
    # Integrator kernels - Runge-Kutta
    'src/integrators/runge_kutta/heun_fused.cu',
    'src/integrators/runge_kutta/heun_backward.cu',
]

# Convert to absolute paths
cuda_sources = [os.path.join(os.path.dirname(__file__), src) for src in cuda_sources]

# Include directories
include_dirs = [
    cuda_src_dir,
    os.path.join(cuda_src_dir, 'common'),
    os.path.join(cuda_src_dir, 'geometry'),
    os.path.join(cuda_src_dir, 'integrators'),
]

setup(
    name='gfn_cuda',
    ext_modules=[
        CUDAExtension(
            name='gfn_cuda',
            sources=cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '--expt-relaxed-constexpr',
                    '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx, T4
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
