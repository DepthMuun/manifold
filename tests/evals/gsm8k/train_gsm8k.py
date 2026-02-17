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
import time
import re

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.optimizers import RiemannianAdam
from gfn.losses import GFNLoss, ToroidalDistanceLoss
from tests.evals.common.manifold_adapter import ManifoldAdapter, OPTIMAL_PHYSICS_CONFIG
from tests.evals.common.viz_engine import VizEngine

console = Console()

class GSM8KTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.out_dir = Path("results/evals/gsm8k")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Dataset
        console.print("[bold cyan]Loading GSM8K from Hugging Face...[/]")
        self.dataset = load_dataset("gsm8k", "main", trust_remote_code=True)
        
        # GPT-2 tokenizer has vocab_size=50257
        # Using 1M was causing CUDA assert failures in cross-entropy loss
        self.adapter = ManifoldAdapter(
            model_path=None,
            device=self.device,
            vocab_size=50257,
            dim=128,
            depth=6,
            heads=4
        )
        self.model = self.adapter.model
        
        # Setup Optimizer (Riemannian with normalization retraction)
        self.optimizer = RiemannianAdam(
            self.model.parameters(), 
            lr=1e-4, 
            retraction='normalize'
        )
        
        # Initialize Loss Functions
        self.gfn_loss = GFNLoss(
            lambda_h=0.001, # Hamiltonian weight
            lambda_g=0.0001, # Geodesic weight
            hamiltonian_mode='adaptive',
            geodesic_mode='structural'
        )
        self.toroidal_loss = ToroidalDistanceLoss()
        
    def extract_answer(self, text):
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if match: return match.group(1).replace(",", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[-1] if numbers else "0"

    def train_step(self, question, answer_text):
        """Training step with full geometric loss stack."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Prepare Inputs & Targets
        prompt = f"Problem: {question}\nSolution: {answer_text}"
        tokens = self.adapter.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        
        # 2. Forward Pass (Collecting Christoffel and Sequences)
        # Returns: logits, state, christoffels, v_seq, x_seq, all_forces
        output = self.model(input_ids, collect_christ=False)
        logits, _, christoffels, v_seq, x_seq, all_forces = output
        
        # 3. Compute Primary GFN Loss (CE + Hamiltonian + Geodesic)
        total_loss, loss_dict = self.gfn_loss(
            logits=logits,
            targets=target_ids,
            velocities=v_seq,
            christoffel_outputs=christoffels,
            states=x_seq,
            forces=all_forces
        )
        
        # 4. Compute Auxiliary Toroidal Distance Loss
        # We supervise the manifold trajectory positions (x_seq) 
        # using the embeddings of the target tokens.
        target_embeddings = self.model.embedding(target_ids)
        t_dist = self.toroidal_loss(x_seq, target_embeddings)
        
        # Combined Loss
        combined_loss = total_loss + 0.1 * t_dist
        
        # 5. Backward & Step
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return combined_loss.item(), loss_dict

    def evaluate(self, num_samples=50):
        self.model.eval()
        correct, total = 0, 0
        test_split = self.dataset['test']
        indices = np.random.choice(len(test_split), min(num_samples, len(test_split)), replace=False)
        
        for idx in tqdm(indices, desc="Evaluating GSM8K"):
            item = test_split[int(idx)]
            pred_answer, _, _, _ = self.adapter.predict_gsm8k(item['question'])
            gold_answer = self.extract_answer(item['answer'])
            
            if pred_answer.strip() == gold_answer.strip():
                correct += 1
            total += 1
            
        return correct / total if total > 0 else 0.0

    def run_suite(self, iterations=100, eval_samples=10):
        history = {"loss": [], "accuracy": [], "ham_loss": [], "geo_loss": []}
        
        # Initial Eval
        acc = self.evaluate(num_samples=eval_samples)
        history["accuracy"].append(acc)
        
        train_split = self.dataset['train']
        pbar = tqdm(range(iterations), desc="Geometric Training (GSM8K)")
        for i in pbar:
            idx = np.random.randint(0, len(train_split))
            item = train_split[idx]
            
            loss_val, ldict = self.train_step(item['question'], item['answer'])
            history["loss"].append(loss_val)
            history["ham_loss"].append(ldict.get("hamiltonian", 0))
            history["geo_loss"].append(ldict.get("geodesic", 0))
            
            pbar.set_postfix({
                "L": f"{loss_val:.3f}", 
                "H": f"{ldict.get('hamiltonian', 0):.4f}",
                "G": f"{ldict.get('geodesic', 0):.4f}"
            })
            
            if (i+1) % 50 == 0:
                acc = self.evaluate(num_samples=eval_samples)
                history["accuracy"].append(acc)
            else:
                history["accuracy"].append(history["accuracy"][-1])
                
        self.create_plots(history)

    def create_plots(self, history):
        fig, axes = VizEngine.create_dashboard("GSM8K Geometric Performance")
        
        # Loss (Top Left)
        VizEngine.plot_curve(axes[0, 0], np.arange(len(history['loss'])), history['loss'], "Total Loss", moving_average=10)
        axes[0, 0].set_title("Training Convergence")
        
        # Physics (Top Right)
        VizEngine.plot_curve(axes[0, 1], np.arange(len(history['ham_loss'])), history['ham_loss'], "Hamiltonian", moving_average=10)
        VizEngine.plot_curve(axes[0, 1], np.arange(len(history['geo_loss'])), history['geo_loss'], "Geodesic", 'secondary', moving_average=10)
        axes[0, 1].set_title("Geometric Regularization")
        axes[0, 1].legend()
        
        # Accuracy (Bottom Left)
        VizEngine.plot_curve(axes[1, 0], np.arange(len(history['accuracy'])), history['accuracy'], "Exact Match", 'tertiary')
        axes[1, 0].set_title("Solving Accuracy")
        
        # Placeholder
        axes[1, 1].set_title("Energy Conservation")
        axes[1, 1].text(0.5, 0.5, "Metric Stability\n(Live Tracking)", ha='center', va='center')
        
        VizEngine.save_dashboard(fig, self.out_dir / "gsm8k_geometric_report.png")
        console.print(f"[bold green]Geometric report saved to {self.out_dir / 'gsm8k_geometric_report.png'}[/]")

if __name__ == "__main__":
    trainer = GSM8KTrainer()
    trainer.run_suite(iterations=100, eval_samples=10)
