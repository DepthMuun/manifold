from gfn.computation.backend.registry import BackendRegistry
import torch

def verify_backend_dispatch():
    cpu_device = torch.device('cpu')
    print(f"CPU device resolved as: {BackendRegistry.get_backend(cpu_device)}")
    
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda:0')
        print(f"CUDA device resolved as: {BackendRegistry.get_backend(cuda_device)}")
        
        t = torch.randn(5, device=cuda_device)
        print(f"Is CUDA active for tensor: {BackendRegistry.is_cuda_active(t)}")
    else:
        print("CUDA not available for testing.")

if __name__ == "__main__":
    verify_backend_dispatch()
