import json
import os
import time
from collections import defaultdict
from typing import Dict, Any, List

from ..common.adapters import BaseAdapter
from ..common.logging_utils import setup_logger, json_log
from ..common.metrics import accuracy, memory_usage_mb, mean_runtime_ms
from ..common.storage import write_jsonl, write_csv
from ..common.seeds import set_global_seed, fix_determinism


class MMLUEvaluator:
    def __init__(self, adapter: BaseAdapter, out_dir: str):
        self.adapter = adapter
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.logger = setup_logger("MMLU", os.path.join(out_dir, "mmlu.log"))
        set_global_seed(1337)
        fix_determinism()

    def run(self, data_path: str) -> Dict[str, Any]:
        with open(data_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        y_pred = []
        y_true = []
        per_cat = defaultdict(lambda: {"pred": [], "true": []})
        results_rows = []
        times = []
        skipped = 0

        def _valid(r):
            if "question" not in r or "choices" not in r or "answer_idx" not in r:
                return False
            if not isinstance(r["choices"], list) or len(r["choices"]) < 2:
                return False
            if not isinstance(r["answer_idx"], int) or not (0 <= r["answer_idx"] < len(r["choices"])):
                return False
            return True

        def _macro_f1(y_t, y_p):
            if not y_t:
                return 0.0
            classes = set(y_t) | set(y_p)
            f1s = []
            for c in classes:
                tp = sum(1 for yt, yp in zip(y_t, y_p) if yt == c and yp == c)
                fp = sum(1 for yt, yp in zip(y_t, y_p) if yt != c and yp == c)
                fn = sum(1 for yt, yp in zip(y_t, y_p) if yt == c and yp != c)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                f1s.append(f1)
            return float(sum(f1s) / len(f1s)) if f1s else 0.0
        for i, r in enumerate(rows):
            t0 = time.time()
            pred_idx, conf, meta = self.adapter.predict_mmlu(r["question"], r["choices"])
            if not _valid(r):
                skipped += 1
                continue
            pred_idx, conf, meta = self.adapter.predict_mmlu(r["question"], r["choices"])
            times.append(dt_ms)

            y_pred.append(pred_idx)
            y_true.append(r["answer_idx"])
            per_cat[r.get("category", "unknown")]["pred"].append(pred_idx)
            per_cat[r.get("category", "unknown")]["true"].append(r["answer_idx"])


            row = {
                "id": i,
                "category": r.get("category", "unknown"),
                "question": r["question"],
                "choices": r["choices"],
                "answer_idx": r["answer_idx"],
                "pred_idx": pred_idx,
                "confidence": conf,
                "latency_ms": round(dt_ms, 3),
            }
            results_rows.append(row)
            json_log(self.logger, "sample", row)

        overall_acc = accuracy(y_pred, y_true)
        macro_f1 = _macro_f1(y_true, y_pred)
        cat_metrics = {c: {"accuracy": accuracy(v["pred"], v["true"]), "n": len(v["pred"])} for c, v in per_cat.items()}
        report = {
            "overall": {"accuracy": overall_acc, "macro_f1": macro_f1, "precision": overall_acc, "recall": overall_acc, "n": len(y_pred), "skipped": skipped, "perplexity": None},
            "categories": cat_metrics,
            "runtime_ms_mean": mean_runtime_ms(times),
            "memory_mb": memory_usage_mb(),
        }

        write_jsonl(os.path.join(self.out_dir, "mmlu_results.jsonl"), results_rows)
        write_csv(os.path.join(self.out_dir, "mmlu_results.csv"), results_rows, ["id", "category", "answer_idx", "pred_idx", "confidence", "latency_ms", "question", "choices"])
        with open(os.path.join(self.out_dir, "mmlu_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        json_log(self.logger, "summary", report)
        return report
