import json
import os
import random
import string
import time
from typing import Dict, Any, List, Tuple

from ..common.adapters import BaseAdapter
from ..common.logging_utils import setup_logger, json_log
from ..common.metrics import accuracy, memory_usage_mb, mean_runtime_ms
from ..common.storage import write_jsonl, write_csv
from ..common.seeds import set_global_seed, fix_determinism


def synth_doc(num_paragraphs: int, words_per_paragraph: int) -> str:
    rng = random.Random(1337)
    words = []
    for _ in range(num_paragraphs):
        para = " ".join("".join(rng.choices(string.ascii_lowercase, k=5)) for _ in range(words_per_paragraph))
        words.append(para)
    return "\n".join(words)


class RULEREvaluator:
    def __init__(self, adapter: BaseAdapter, out_dir: str):
        self.adapter = adapter
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.logger = setup_logger("RULER", os.path.join(out_dir, "ruler.log"))
        set_global_seed(1337)
        fix_determinism()

    def run(self, docs: List[Tuple[str, str]]) -> Dict[str, Any]:
        preds = []
        trues = []
        times = []
        rows = []
        for i, (context, question) in enumerate(docs):
            t0 = time.time()
            ans, conf, meta = self.adapter.retrieve_from_context(context, question)
            dt_ms = (time.time() - t0) * 1000.0
            times.append(dt_ms)
            ok = "NEEDLE:" in context and (ans in context)
            trues.append(1 if ok else 0)
            preds.append(1 if ok else 0)
            row = {
                "id": i,
                "ctx_len": len(context),
                "question": question,
                "pred_nonempty": int(len(ans) > 0),
                "confidence": conf,
                "latency_ms": round(dt_ms, 3),
                "retrieved": ans,
            }
            rows.append(row)
            json_log(self.logger, "sample", row)

        report = {
            "retention_proxy_acc": accuracy(preds, trues),
            "n": len(rows),
            "runtime_ms_mean": mean_runtime_ms(times),
            "memory_mb": memory_usage_mb(),
        }
        write_jsonl(os.path.join(self.out_dir, "ruler_results.jsonl"), rows)
        write_csv(os.path.join(self.out_dir, "ruler_results.csv"), rows, ["id", "ctx_len", "pred_nonempty", "confidence", "latency_ms", "question", "retrieved"])
        with open(os.path.join(self.out_dir, "ruler_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        json_log(self.logger, "summary", report)
        return report
