from typing import Tuple, Dict, Any, List
import random

import torch


class BaseAdapter:
    def predict_mmlu(self, question: str, choices: List[str]) -> Tuple[int, float, Dict[str, Any]]:
        raise NotImplementedError

    def predict_gsm8k(self, question: str) -> Tuple[str, List[str], float, Dict[str, Any]]:
        raise NotImplementedError

    def retrieve_from_context(self, context: str, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        raise NotImplementedError


class RandomAdapter(BaseAdapter):
    def predict_mmlu(self, question: str, choices: List[str]) -> Tuple[int, float, Dict[str, Any]]:
        idx = random.randrange(len(choices))
        return idx, 0.25, {"strategy": "random"}

    def predict_gsm8k(self, question: str) -> Tuple[str, List[str], float, Dict[str, Any]]:
        pred = "0"
        steps = ["guess 0"]
        return pred, steps, 0.0, {"strategy": "constant_zero"}

    def retrieve_from_context(self, context: str, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        return "", 0.0, {"strategy": "none"}


class OracleAdapter(BaseAdapter):
    def predict_mmlu(self, question: str, choices: List[str]) -> Tuple[int, float, Dict[str, Any]]:
        return 0, 1.0, {"strategy": "oracle_placeholder"}

    def predict_gsm8k(self, question: str) -> Tuple[str, List[str], float, Dict[str, Any]]:
        return "1", ["read question", "answer 1"], 1.0, {"strategy": "oracle_placeholder"}

    def retrieve_from_context(self, context: str, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        marker = "NEEDLE:"
        i = context.find(marker)
        if i >= 0:
            j = context.find("\n", i)
            needle = context[i + len(marker): j if j > 0 else None].strip()
            return needle, 1.0, {"strategy": "needle_substring"}
        return "", 0.0, {"strategy": "needle_not_found"}
