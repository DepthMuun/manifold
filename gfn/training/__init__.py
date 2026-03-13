"""
gfn/training/__init__.py
Public API del módulo training — GFN V5
"""

from gfn.training.trainer import GFNTrainer
from gfn.training.optimizer import (
    RiemannianAdam, RiemannianSGD, create_optimizer,
    make_gfn_optimizer, all_parameters,
)
from gfn.training.scheduler import WarmupCosineScheduler, create_scheduler
from gfn.training.metrics import accuracy, perplexity, last_token_accuracy, compute_metrics
from gfn.training.callbacks import Callback
from gfn.training.callbacks.checkpoint import CheckpointCallback
from gfn.training.callbacks.early_stopping import EarlyStoppingCallback
from gfn.training.callbacks.logger import LoggerCallback
from gfn.training.checkpoint import save_checkpoint, load_checkpoint

__all__ = [
    "GFNTrainer",
    # Optimizers
    "RiemannianAdam", "RiemannianSGD", "create_optimizer",
    "make_gfn_optimizer", "all_parameters",
    # Schedulers
    "WarmupCosineScheduler", "create_scheduler",
    # Metrics
    "accuracy", "perplexity", "last_token_accuracy", "compute_metrics",
    # Callbacks
    "Callback", "CheckpointCallback", "EarlyStoppingCallback", "LoggerCallback",
    # Checkpoint
    "save_checkpoint", "load_checkpoint",
]
