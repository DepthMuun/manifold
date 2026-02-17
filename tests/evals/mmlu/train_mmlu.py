import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from rich.console import Console

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.optimizers import RiemannianAdam
from gfn.losses import GFNLoss, ToroidalDistanceLoss
from tests.evals.common.manifold_adapter import ManifoldAdapter, OPTIMAL_PHYSICS_CONFIG
from tests.evals.common.viz_engine import VizEngine

console = Console()

class MMLUTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.out_dir = Path("results/evals/mmlu")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Adapter
        # GPT-2 tokenizer has vocab_size=50257
        # Using 10M was creating a ~5GB embedding and causing CUDA assert failures
        self.adapter = ManifoldAdapter(
            model_path=None,
            device=self.device,
            vocab_size=50257,
            dim=128,
            depth=6,
            heads=4
        )
        self.model = self.adapter.model
        
        # Riemannian Optimizer
        self.optimizer = RiemannianAdam(self.model.parameters(), lr=1e-4, retraction='normalize')
        
        # Geometric Losses
        self.gfn_loss = GFNLoss(lambda_h=0.001, lambda_g=0.0001)
        self.toroidal_loss = ToroidalDistanceLoss()

    def train_step(self, question, choices, answer_idx):
        """Train step with geometric supervision for multiple choice."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Format MCQ Prompt
        labels = ["A", "B", "C", "D"]
        prompt = f"Question: {question}\n"
        for i, c in enumerate(choices):
            prompt += f"{labels[i]}) {c}\n"
        prompt += f"Answer: {labels[answer_idx]}"
        
        tokens = self.adapter.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        
        # 2. Forward
        output = self.model(input_ids, collect_christ=True)
        logits, _, christs, v_seq, x_seq, forces = output
        
        # 3. Compute Losses
        ce_loss, ldict = self.gfn_loss(logits, target_ids, v_seq, christs, states=x_seq, forces=forces)
        target_embs = self.model.embedding(target_ids)
        t_dist = self.toroidal_loss(x_seq, target_embs)
        
        total = ce_loss + 0.1 * t_dist
        total.backward()
        self.optimizer.step()
        
        return total.item(), ldict

    def run_suite(self, subjects=["college_physics", "professional_law"], iterations=50):
        history = {"loss": [], "accuracy": [], "ham_loss": []}
        
        for sub in subjects:
            console.print(f"[cyan]Geometric Training on MMLU Subject: {sub}[/]")
            ds = load_dataset("cais/mmlu", sub, split="test", trust_remote_code=True)
            pbar = tqdm(range(iterations), desc=f"MMLU: {sub}")
            
            for i in pbar:
                idx = np.random.randint(0, len(ds))
                item = ds[idx]
                loss, ldict = self.train_step(item['question'], item['choices'], item['answer'])
                
                history["loss"].append(loss)
                history["ham_loss"].append(ldict.get("hamiltonian", 0))
                pbar.set_postfix({"L": f"{loss:.3f}", "H": f"{ldict.get('hamiltonian', 0):.4f}"})

        self.create_plots(history)

    def create_plots(self, history):
        fig, axes = VizEngine.create_dashboard("MMLU Geometric Fine-Tuning")
        VizEngine.plot_curve(axes[0, 0], np.arange(len(history['loss'])), history['loss'], "Combined Loss")
        VizEngine.plot_curve(axes[0, 1], np.arange(len(history['ham_loss'])), history['ham_loss'], "Hamiltonian", 'secondary')
        axes[1, 0].set_title("Manifold Density")
        axes[1, 1].set_title("VRAM Consumption")
        VizEngine.save_dashboard(fig, self.out_dir / "mmlu_geometric_report.png")

if __name__ == "__main__":
    trainer = MMLUTrainer()
    trainer.run_suite()
