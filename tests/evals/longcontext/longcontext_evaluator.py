"""
LongContext Evaluator for Manifold Model
Professional implementation using NIAH (Needle in a Haystack) protocol.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.evals.common.common_evaluator import BaseEvaluator

class LongContextEvaluator(BaseEvaluator):
    def __init__(self, device: str = None):
        super().__init__("LongContext", device)
        self.logger.info("Initializing streaming haystack (WikiText-103)...")
        self.dataset = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=True, trust_remote_code=True)
        self.haystack_iter = iter(self.dataset)

    def create_test_case(self, length: int, depth: float):
        """Inject needle into haystack at specified depth."""
        hay_text = ""
        while len(self.adapter.tokenizer.encode(hay_text)) < length:
            try:
                hay_text += next(self.haystack_iter)['text'] + " "
            except StopIteration:
                break
                
        tokens = self.adapter.tokenizer.encode(hay_text)[:length]
        needle = "The secret passkey is 'GFN-99-MANIFOLD'."
        needle_tokens = self.adapter.tokenizer.encode(needle)
        
        pos = int((length - len(needle_tokens)) * (depth / 100.0))
        test_tokens = tokens[:pos] + needle_tokens + tokens[pos + len(needle_tokens):]
        test_tokens = test_tokens[:length]
        
        question = "\nWhat is the secret passkey?\nAnswer:"
        q_tokens = self.adapter.tokenizer.encode(question, add_special_tokens=False)
        
        return torch.tensor([test_tokens + q_tokens]).to(self.device), "GFN-99-MANIFOLD"

    def run(self, lengths: list = [1024, 2048, 4096], depths: list = [0, 50, 100]):
        """Run NIAH grid evaluation."""
        self.model.eval()
        results = np.zeros((len(lengths), len(depths)))
        
        for i, l in enumerate(lengths):
            for j, d in enumerate(depths):
                self.logger.info(f"Testing NIAH: Length={l}, Depth={d}%")
                input_ids, expected = self.create_test_case(l, d)
                
                with torch.no_grad():
                    generated = []
                    curr = input_ids
                    for _ in range(10):
                        out = self.model(curr)
                        logits = out[0] if isinstance(out, tuple) else out
                        nxt = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                        generated.append(nxt.item())
                        curr = torch.cat([curr, nxt], dim=1)
                        if nxt.item() == self.adapter.tokenizer.eos_token_id:
                            break
                            
                pred = self.adapter.tokenizer.decode(generated)
                results[i, j] = 1.0 if expected in pred else 0.0
                
        metrics = {
            "avg_recall": results.mean(),
            "grid": results.tolist(),
            "lengths": lengths,
            "depths": depths
        }
        
        self.save_results(metrics)
        self.generate_dashboard({"accuracy": [results.mean()]})
        return metrics

if __name__ == "__main__":
    evaluator = LongContextEvaluator()
    evaluator.run()