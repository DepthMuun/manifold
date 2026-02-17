"""
MMLU Evaluator for Manifold Model
Professional implementation using Hugging Face datasets and log-probability scoring.
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

class MMLUEvaluator(BaseEvaluator):
    def __init__(self, device: str = None):
        super().__init__("MMLU", device)

    def get_log_probs(self, question: str, choices: list) -> int:
        """Calculate log-probabilities for choices and return predicted index."""
        choice_labels = ["A", "B", "C", "D"]
        choice_token_ids = [self.adapter.tokenizer.encode(lbl, add_special_tokens=False)[0] for lbl in choice_labels]
        
        prompt = f"Question: {question}\n"
        for i, c in enumerate(choices):
            prompt += f"{choice_labels[i]}) {c}\n"
        prompt += "Answer:"
        
        input_ids = self.adapter.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids)
            logits = output[0] if isinstance(output, tuple) else output
            last_token_logits = logits[0, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)
            
            lprobs = [torch.log(probs[tid]).item() for tid in choice_token_ids]
            return np.argmax(lprobs)

    def run(self, subjects: list = ["college_physics", "professional_law"], num_samples_per_sub: int = 10):
        """Run evaluation across multiple MMLU subjects."""
        self.model.eval()
        overall_history = {"accuracy": []}
        subject_results = {}
        
        for sub in subjects:
            self.logger.info(f"Evaluating MMLU subject: {sub}")
            ds = load_dataset("cais/mmlu", sub, split="test", trust_remote_code=True)
            correct = 0
            total = 0
            
            indices = np.random.choice(len(ds), min(num_samples_per_sub, len(ds)), replace=False)
            for idx in indices:
                item = ds[int(idx)]
                pred_idx = self.get_log_probs(item['question'], item['choices'])
                if pred_idx == item['answer']:
                    correct += 1
                total += 1
            
            acc = correct / total if total > 0 else 0.0
            subject_results[sub] = acc
            overall_history["accuracy"].append(acc)
            
        metrics = {
            "overall_accuracy": np.mean(list(subject_results.values())),
            "subject_breakdown": subject_results
        }
        
        self.save_results(metrics)
        # Placeholder for specialized MMLU plots
        self.generate_dashboard(overall_history)
        return metrics

if __name__ == "__main__":
    evaluator = MMLUEvaluator()
    evaluator.run()