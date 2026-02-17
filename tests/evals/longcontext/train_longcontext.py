import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from rich.console import Console

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.optimizers import RiemannianAdam
from gfn.losses import GFNLoss, ToroidalDistanceLoss
from tests.evals.common.manifold_adapter import ManifoldAdapter
from tests.evals.common.viz_engine import VizEngine

console = Console()

class LongContextTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.out_dir = Path("results/evals/longcontext")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Haystack Dataset (WikiText-103 is robust for long-context testing)
        console.print("[bold cyan]Loading Haystack Dataset (WikiText-103)...[/]")
        try:
            # wikitext-103-v1 provides long enough articles for the 'hay'
            self.dataset = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=True, trust_remote_code=True)
            self.haystack_iter = iter(self.dataset)
        except Exception as e:
            console.print(f"[red]Error loading dataset: {e}[/]")
            self.dataset = None
        
        self.adapter = ManifoldAdapter(None, vocab_size=50257, dim=768, depth=12, device=device)
        self.model = self.adapter.model
        self.optimizer = RiemannianAdam(self.model.parameters(), lr=1e-4, retraction='normalize')
        
        # Physics-intensive loss for long context (higher lambda_g to prevent drift)
        self.gfn_loss = GFNLoss(lambda_h=0.005, lambda_g=0.0005)
        self.toroidal_loss = ToroidalDistanceLoss()

    def train_step(self, text_chunk):
        """Train step for sequence length stability."""
        self.model.train()
        self.optimizer.zero_grad()
        
        tokens = self.adapter.tokenizer.encode(text_chunk, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        
        output = self.model(input_ids, collect_christ=True)
        logits, _, christs, v_seq, x_seq, forces = output
        
        loss, ldict = self.gfn_loss(logits, target_ids, v_seq, christs, states=x_seq, forces=forces)
        target_embs = self.model.embedding(target_ids)
        t_dist = self.toroidal_loss(x_seq, target_embs)
        
        total = loss + 0.2 * t_dist
        total.backward()
        self.optimizer.step()
        
        return total.item(), ldict

    def run_suite(self, iterations=30):
        history = {"loss": [], "ham_loss": [], "geo_loss": []}
        
        for i in tqdm(range(iterations), desc="LongContext Stability"):
            try:
                item = next(self.haystack_iter)
                loss, ldict = self.train_step(item['text'][:5000])
                history["loss"].append(loss)
                history["ham_loss"].append(ldict.get("hamiltonian", 0))
                history["geo_loss"].append(ldict.get("geodesic", 0))
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                
        self.create_plots(history)

    def create_plots(self, history):
        fig, axes = VizEngine.create_dashboard("LongContext Geometric Stability")
        VizEngine.plot_curve(axes[0, 0], np.arange(len(history['loss'])), history['loss'], "Loss")
        VizEngine.plot_curve(axes[0, 1], np.arange(len(history['ham_loss'])), history['ham_loss'], "Energy Error", 'secondary')
        axes[1, 0].set_title("Curvature Variance")
        axes[1, 1].set_title("Gradient Norms")
        VizEngine.save_dashboard(fig, self.out_dir / "longcontext_geometric_report.png")

if __name__ == "__main__":
    trainer = LongContextTrainer()
    trainer.run_suite()
