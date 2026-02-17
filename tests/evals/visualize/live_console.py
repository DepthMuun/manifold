import argparse
import json
import os
import time
from typing import Dict, Any


def aggregate_jsonl(path: str) -> Dict[str, Any]:
    stats = {"n": 0, "ok": 0, "latency_sum": 0.0}
    if not os.path.exists(path):
        return stats
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            stats["n"] += 1
            if "ok" in r:
                stats["ok"] += int(bool(r["ok"]))
            if "pred_idx" in r and "answer_idx" in r:
                stats["ok"] += int(r["pred_idx"] == r["answer_idx"])
            stats["latency_sum"] += float(r.get("latency_ms", 0.0))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to JSONL results")
    ap.add_argument("--interval", type=float, default=1.0, help="Refresh interval seconds")
    args = ap.parse_args()

    try:
        while True:
            s = aggregate_jsonl(args.file)
            n = s.get("n", 0)
            ok = s.get("ok", 0)
            acc = (ok / n) if n else 0.0
            lat = (s.get("latency_sum", 0.0) / n) if n else 0.0
            print(f"[live] n={n} ok={ok} acc={acc:.3f} latency_ms={lat:.3f}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
