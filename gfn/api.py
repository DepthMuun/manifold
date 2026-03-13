"""
gfn/api.py — GFN V5
Interfaz pública simplificada y orquestación de alto nivel.
Centraliza la creación, carga y evaluación de modelos.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from gfn.models.factory import ModelFactory
from gfn.losses.factory import LossFactory
from gfn.training.trainer import GFNTrainer
from gfn.training.evaluation import ManifoldMetricEvaluator

# -- Alias principales
Trainer = GFNTrainer

def create(*args, **kwargs):
    """Factory para modelos Manifold (V5)."""
    return ModelFactory.create(*args, **kwargs)

def loss(config, **kwargs):
    """Factory para funciones de pérdida (V5)."""
    return LossFactory.create(config, **kwargs)

def save(model: nn.Module, path: str):
    """
    Guarda el modelo y su configuración (HuggingFace Style).
    """
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(path)
    else:
        # Fallback para modelos que no heredan de BaseModel
        torch.save({'state_dict': model.state_dict()}, path)

def load(path: str, device: Optional[str] = None):
    """
    Carga un modelo guardado junto con su configuración.
    Soporta directorios (HF Style) o archivos .pth/.bin legados.
    """
    import os
    if os.path.isdir(path):
        return ModelFactory.from_pretrained(path)
    
    # Fallback para archivos aislados legados
    checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=True)
    config = checkpoint.get('config')
    if config is None:
        raise ValueError(f"No se encontró configuración en el checkpoint {path}. Use directorios HF para carga completa.")
        
    model = create(config=config)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def benchmark(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
              device: Optional[str] = None) -> Dict[str, float]:
    """
    Ejecuta una evaluación rápida de métricas geométricas y de tarea.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    evaluator = ManifoldMetricEvaluator(model)
    all_x, all_v, all_y = [], [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, (xf, vf), info = model(x)
            
            all_x.append(xf.detach().cpu())
            all_v.append(vf.detach().cpu())
            all_y.append(y.detach().cpu())
            
    if not all_x:
        return {}

    x_total = torch.cat(all_x, dim=0)
    v_total = torch.cat(all_v, dim=0)
    y_total = torch.cat(all_y, dim=0)
    
    return evaluator.full_report(x_total, v_total, y_total)
