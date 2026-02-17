import json
import os
import time
from typing import Dict, Any, List

from ..common.adapters import BaseAdapter
from ..common.logging_utils import setup_logger, json_log
from ..common.metrics import accuracy, memory_usage_mb, mean_runtime_ms
from ..common.storage import write_jsonl, write_csv
from ..common.seeds import set_global_seed, fix_determinism


def normalize_numeric(s: str) -> str:
    try:
        return str(int(float(s.strip())))
    except Exception:
        return s.strip()


class GSM8KEvaluator:
    def __init__(self, adapter: BaseAdapter, out_dir: str):
        self.adapter = adapter
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.logger = setup_logger("GSM8K", os.path.join(out_dir, "gsm8k.log"))
        set_global_seed(1337)
        fix_determinism()

    def run(self, data_path: str) -> Dict[str, Any]:
        with open(data_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        correct = 0
        total = 0
        times = []
        per_type = {"format_error": 0, "numeric_mismatch": 0, "other": 0}
        result_rows: List[Dict[str, Any]] = []

        skipped = 0
        for i, r in enumerate(rows):
            if "question" not in r or "answer" not in r:
                skipped += 1
                continue
            t0 = time.time()
            pred, steps, conf, meta = self.adapter.predict_gsm8k(r["question"])
            dt_ms = (time.time() - t0) * 1000.0
            times.append(dt_ms)
            total += 1

            gt = normalize_numeric(r["answer"])
            pd = normalize_numeric(pred)
            ok = (gt == pd)
            if ok:
                correct += 1
            else:
                try:
                    float(pd)
                except Exception:
                    per_type["format_error"] += 1
                else:
                    per_type["numeric_mismatch"] += 1

            row = {
                "id": i,
                "question": r["question"],
                "answer": r["answer"],
                "pred": pred,
                "confidence": conf,
                "steps": steps,
                "steps_len": len(steps),
                "latency_ms": round(dt_ms, 3),
                "ok": ok,
            }
            result_rows.append(row)
            json_log(self.logger, "sample", row)

        acc = correct / total if total else 0.0
        report = {
            "accuracy": acc,
            "precision": acc,
            "recall": acc,
            "f1": acc,
            "perplexity": None,
            "n": total,
            "skipped": skipped,
            "error_types": per_type,
            "runtime_ms_mean": mean_runtime_ms(times),
            "memory_mb": memory_usage_mb(),
        }

        write_jsonl(os.path.join(self.out_dir, "gsm8k_results.jsonl"), result_rows)
        write_csv(os.path.join(self.out_dir, "gsm8k_results.csv"), result_rows, ["id", "ok", "confidence", "steps_len", "latency_ms", "question", "answer", "pred", "steps"])
        with open(os.path.join(self.out_dir, "gsm8k_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        json_log(self.logger, "summary", report)
        return report
