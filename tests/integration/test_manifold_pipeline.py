import os
import time
import unittest
import json
import torch

from tests.manifold_system.pipeline import build_default_config, ModelConfigurator, AnalysisRunner, DataValidator, ReportFormatter


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.out_dir = os.path.join(os.getcwd(), "_manifold_sys_integration")
        self.cfg = build_default_config()
        self.model = ModelConfigurator.create_model(16, 48, 2, 2, "heun", self.cfg)

    def test_end_to_end(self):
        forces = torch.zeros(1, 6, self.model.dim)
        r = AnalysisRunner(self.out_dir)
        rep = r.run(self.model, forces, collect=False)
        self.assertTrue(rep["summary"]["ok"])
        path = os.path.join(self.out_dir, "report.json")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertIn("sections", loaded)

    def test_edge_cases(self):
        forces = torch.randn(1, 1, self.model.dim) * 0.01
        DataValidator.validate_forces(forces, self.model.dim)
        r = AnalysisRunner(self.out_dir)
        rep = r.run(self.model, forces, collect=False)
        self.assertTrue(rep["summary"]["ok"])

    def test_robust_inputs(self):
        forces = torch.randn(2, 4, self.model.dim)
        forces[0, 0, 0] = 0.0
        r = AnalysisRunner(self.out_dir)
        rep = r.run(self.model, forces, collect=False)
        self.assertTrue(rep["summary"]["ok"])

    def test_performance_scaled(self):
        forces = torch.randn(4, 32, self.model.dim) * 0.02
        r = AnalysisRunner(self.out_dir)
        t0 = time.time()
        r.run(self.model, forces, collect=False)
        dt = (time.time() - t0) * 1000.0
        self.assertLess(dt, 8000.0)


if __name__ == "__main__":
    unittest.main()