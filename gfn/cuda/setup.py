from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Build path
cuda_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='gfn_cuda',
    ext_modules=[
        CUDAExtension(
            'gfn_cuda',
            [
                'cuda_kernels.cpp',
                'src/geometry/christoffel_fused.cu',
                'src/geometry/lowrank_christoffel.cu',
                'src/geometry/reactive_christoffel.cu',
                'src/integrators/symplectic/leapfrog_fused.cu',
                'src/integrators/symplectic/leapfrog_backward.cu',
                'src/integrators/symplectic/yoshida_fused.cu',
                'src/integrators/runge_kutta/euler_fused.cu',
                'src/integrators/symplectic/verlet_fused.cu',
                'src/integrators/symplectic/forest_ruth_fused.cu',
                'src/integrators/symplectic/omelyan_fused.cu',
                'src/integrators/runge_kutta/heun_fused.cu',
                'src/integrators/runge_kutta/rk4_fused.cu',
                'src/integrators/runge_kutta/dormand_prince_fused.cu',
                'src/integrators/recurrent/recurrent_manifold_fused.cu',
                'src/integrators/recurrent/recurrent_manifold_backward.cu',
                'src/integrators/recurrent/manifold_step.cu',
                'src/layers/parallel_scan_fused.cu',
                'src/layers/head_mixing.cu',
                'src/layers/dynamic_gating.cu',
            ],
            include_dirs=[os.path.join(cuda_dir, 'include')],
            extra_compile_args={
                'cxx': ['/std:c++17', '/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/permissive-', '/Zc:__cplusplus', '/Zm2000', '/wd4996'],
                'nvcc': [
                    '-O2', '--use_fast_math', '-std=c++17',
                    '-Xcompiler', '/std:c++17', 
                    '-Xcompiler', '/DNOMINMAX',
                    '-Xcompiler', '/DWIN32_LEAN_AND_MEAN',
                    '-Xcompiler', '/permissive-',
                    '-Xcompiler', '/Zc:__cplusplus',
                    '-Xcompiler', '/wd4996'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
