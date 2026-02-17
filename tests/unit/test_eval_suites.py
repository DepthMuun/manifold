import os
from tests.evals.common.adapters import RandomAdapter, OracleAdapter
from tests.evals.mmlu.evaluator import MMLUEvaluator
from tests.evals.gsm8k.evaluator import GSM8KEvaluator
from tests.evals.longcontext.ruler import RULEREvaluator, synth_doc
from tests.evals.longcontext.needle import NeedleEvaluator, inject_needle


def test_mmlu_sample_end_to_end(tmp_path):
    adapter = RandomAdapter()
    out_dir = os.path.join(tmp_path, "mmlu")
    ev = MMLUEvaluator(adapter, out_dir)
    sample = os.path.join(os.path.dirname(__file__), "..", "data", "mmlu_sample.jsonl")
    rep = ev.run(os.path.abspath(sample))
    assert "overall" in rep and "accuracy" in rep["overall"]


def test_gsm8k_sample_end_to_end(tmp_path):
    adapter = RandomAdapter()
    out_dir = os.path.join(tmp_path, "gsm8k")
    ev = GSM8KEvaluator(adapter, out_dir)
    sample = os.path.join(os.path.dirname(__file__), "..", "data", "gsm8k_sample.jsonl")
    rep = ev.run(os.path.abspath(sample))
    assert "accuracy" in rep and rep["n"] >= 2


def test_long_context_evaluators(tmp_path):
    adapter = OracleAdapter()
    out_dir_r = os.path.join(tmp_path, "ruler")
    out_dir_n = os.path.join(tmp_path, "needle")

    r = RULEREvaluator(adapter, out_dir_r)
    doc = synth_doc(2, 10) + "\nNEEDLE:token123\n"
    rep_r = r.run([(doc, "Where is the token?")])
    assert "retention_proxy_acc" in rep_r

    n = NeedleEvaluator(adapter, out_dir_n)
    haystack = "a" * 1000
    rep_n = n.run([(haystack, "secret", 500)])
    assert "recovery_accuracy" in rep_n***
