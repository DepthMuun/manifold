import json
import os
import random
import time
from typing import Dict, Any, List, Tuple

from ..common.adapters import BaseAdapter
from ..common.logging_utils import setup_logger, json_log
from ..common.metrics import accuracy, memory_usage_mb, mean_runtime_ms
from ..common.storage import write_jsonl, write_csv
from ..common.seeds import set_global_seed, fix_determinism


def inject_needle(haystack: str, needle: str, pos: int) -> str:
    pos = max(0, min(len(haystack), pos))
    return haystack[:pos] + f"\nNEEDLE:{needle}\n" + haystack[pos:]


class NeedleEvaluator:
    def __init__(self, adapter: BaseAdapter, out_dir: str):
        self.adapter = adapter
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.logger = setup_logger("NEEDLE", os.path.join(out_dir, "needle.log"))
        set_global_seed(1337)
        fix_determinism()

    def run(self, tasks: List[Tuple[str, str, int]]) -> Dict[str, Any]:
        preds = []
        trues = []
        times = []
        rows = []
        for i, (haystack, needle, pos) in enumerate(tasks):
            ctx = inject_needle(haystack, needle, pos)
            prompt = f"Find the hidden key"
            t0 = time.time()
            ans, conf, meta = self.adapter.retrieve_from_context(ctx, prompt)
            dt_ms = (time.time() - t0) * 1000.0
            times.append(dt_ms)
            ok = (ans.strip() == needle.strip())
            preds.append(1 if ok else 0)
            trues.append(1)
            row = {
                "id": i,
                "ctx_len": len(ctx),
                "needle": needle,
                "pos": pos,
                "ok": ok,
                "confidence": conf,
                "latency_ms": round(dt_ms, 3),
                "retrieved": ans,
            }
            rows.append(row)
            json_log(self.logger, "sample", row)

        report = {
            "recovery_accuracy": accuracy(preds, trues),
            "n": len(rows),
            "runtime_ms_mean": mean_runtime_ms(times),
            "memory_mb": memory_usage_mb(),
        }
        write_jsonl(os.path.join(self.out_dir, "needle_results.jsonl"), rows)
        write_csv(os.path.join(self.out_dir, "needle_results.csv"), rows, ["id", "ok", "confidence", "latency_ms", "ctx_len", "pos", "needle", "retrieved"])
        with open(os.path.join(self.out_dir, "needle_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        json_log(self.logger, "summary", report)
        return report
