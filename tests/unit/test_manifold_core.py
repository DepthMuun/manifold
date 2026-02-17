import json
import os
import time
import types
import unittest

import torch

from tests.manifold_system.pipeline import build_default_config, ModelConfigurator, DataValidator, AnalysisRunner, ReportFormatter, run_analysis_example


class TestDataValidator(unittest.TestCase):
    def test_validate_ok(self):
        dim = 16
        forces = torch.randn(2, 3, dim)
        DataValidator.validate_forces(forces, dim)

    def test_validate_dim_mismatch(self):
        dim = 16
        forces = torch.randn(2, 3, dim + 1)
        with self.assertRaises(ValueError):
            DataValidator.validate_forces(forces, dim)

    def test_validate_non_finite(self):
        dim = 8
        forces = torch.randn(1, 2, dim)
        forces[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            DataValidator.validate_forces(forces, dim)

    def test_validate_rank(self):
        dim = 8
        forces = torch.randn(2, dim)
        with self.assertRaises(ValueError):
            DataValidator.validate_forces(forces, dim)


class TestModelConfigurator(unittest.TestCase):
    def test_create_model(self):
        cfg = build_default_config()
        m = ModelConfigurator.create_model(vocab_size=16, dim=32, depth=1, heads=2, integrator="heun", physics_config=cfg)
        self.assertEqual(m.dim, 32)
        self.assertEqual(len(m.layers), 1)


class TestReportFormatter(unittest.TestCase):
    def test_hierarchical_structure(self):
        sec = ReportFormatter.section("H", {"a": 1}, [ReportFormatter.table("T", ["c1"], [[1]])])
        rep = ReportFormatter.hierarchical("Title", {"ok": True}, [sec])
        self.assertIn("sections", rep)
        self.assertEqual(rep["sections"][0]["tables"][0]["columns"], ["c1"])


class TestAnalysisRunner(unittest.TestCase):
    def test_run_small(self):
        out_dir = os.path.join(os.getcwd(), "_manifold_sys_out_unit")
        cfg = build_default_config()
        model = ModelConfigurator.create_model(16, 32, 1, 2, "heun", cfg)
        forces = torch.randn(1, 4, model.dim) * 0.05
        r = AnalysisRunner(out_dir)
        rep = r.run(model, forces)
        self.assertTrue(rep["summary"]["ok"])
        self.assertIn("sections", rep)

    def test_performance_threshold(self):
        out_dir = os.path.join(os.getcwd(), "_manifold_sys_perf")
        cfg = build_default_config()
        model = ModelConfigurator.create_model(16, 32, 2, 2, "heun", cfg)
        forces = torch.randn(2, 16, model.dim) * 0.05
        r = AnalysisRunner(out_dir)
        t0 = time.time()
        r.run(model, forces)
        dt = (time.time() - t0) * 1000.0
        self.assertLess(dt, 3000.0)


class TestExample(unittest.TestCase):
    def test_example_runs(self):
        out_dir = os.path.join(os.getcwd(), "_manifold_example")
        rep = run_analysis_example(out_dir)
        self.assertTrue(rep["summary"]["ok"])


def load_tests(loader, tests, pattern):
    return tests


if __name__ == "__main__":
    unittest.main()