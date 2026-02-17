"""
Common Evaluator Base for Manifold Model
Standardized implementation for production-ready cross-dataset benchmarking.
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.evals.common.manifold_adapter import ManifoldAdapter, OPTIMAL_PHYSICS_CONFIG
from tests.evals.common.viz_engine import VizEngine

class BaseEvaluator(ABC):
    """
    Base class for all Manifold evaluators.
    Provides standardized initialization, adapter management, and visualization.
    """
    def __init__(self, task_name: str, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.task_name = task_name
        self.out_dir = Path(f"results/evals/{task_name.lower()}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Adapter
        self.adapter = self._setup_adapter()
        self.model = self.adapter.model
        
        # Setup Logging
        self.logger = logging.getLogger(f"ManifoldEval_{task_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _setup_adapter(self) -> ManifoldAdapter:
        """Standardized adapter setup."""
        return ManifoldAdapter(
            model_path=None,  # Defaults to new model if not specified
            device=self.device,
            physics_config=OPTIMAL_PHYSICS_CONFIG
        )

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the evaluation suite."""
        pass

    def save_results(self, metrics: Dict[str, Any], filename: str = "results.json"):
        """Save metrics to JSON."""
        import json
        path = self.out_dir / filename
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Results saved to {path}")

    def generate_dashboard(self, history: Dict[str, List[float]]):
        """Invoke VizEngine to create standardized dashboard."""
        fig, axes = VizEngine.create_dashboard(f"{self.task_name} Production Report")
        
        # Default distribution of plots
        if "loss" in history:
            VizEngine.plot_curve(axes[0, 0], np.arange(len(history['loss'])), history['loss'], "Loss", moving_average=10)
            axes[0, 0].set_title("Training Stability")
            
        if "accuracy" in history:
            VizEngine.plot_curve(axes[0, 1], np.arange(len(history['accuracy'])), history['accuracy'], "Accuracy", color_key='secondary')
            axes[0, 1].set_title("Success Rate (EM/Recall)")
            
        # Optional placeholders for specialized metrics
        axes[1, 0].set_title("Physics Metric A")
        axes[1, 1].set_title("Physics Metric B")
        
        VizEngine.save_dashboard(fig, self.out_dir / "dashboard.png")
        self.logger.info(f"Dashboard generated at {self.out_dir / 'dashboard.png'}")