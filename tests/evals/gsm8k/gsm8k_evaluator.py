"""
GSM8K Evaluator for Manifold Model
Professional implementation using Hugging Face datasets and real greedy decoding.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import re

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.evals.common.common_evaluator import BaseEvaluator

class GSM8KEvaluator(BaseEvaluator):
    def __init__(self, device: str = None):
        super().__init__("GSM8K", device)
        self.logger.info("Loading GSM8K dataset...")
        self.dataset = load_dataset("gsm8k", "main", trust_remote_code=True)['test']

    def extract_answer(self, text: str) -> str:
        """Extract numeric answer after #### marker."""
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if match:
            return match.group(1).replace(",", "")
        return "0"

    def run(self, num_samples: int = 50):
        """Run evaluation on the test set."""
        self.model.eval()
        correct = 0
        total = 0
        history = {"accuracy": [], "latency": []}
        
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for idx in tqdm(indices, desc="Evaluating GSM8K"):
            item = self.dataset[int(idx)]
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            pred_answer, reasoning, conf, meta = self.adapter.predict_gsm8k(item['question'])
            end_time.record()
            
            torch.cuda.synchronize()
            latency = start_time.elapsed_time(end_time)
            
            gold_answer = self.extract_answer(item['answer'])
            is_correct = (pred_answer.strip() == gold_answer.strip())
            
            if is_correct:
                correct += 1
            total += 1
            
            history["accuracy"].append(correct / total)
            history["latency"].append(latency)
            
        metrics = {
            "accuracy": correct / total,
            "avg_latency_ms": np.mean(history["latency"]),
            "total_samples": total
        }
        
        self.save_results(metrics)
        self.generate_dashboard(history)
        return metrics

if __name__ == "__main__":
    evaluator = GSM8KEvaluator()
    evaluator.run(num_samples=20)