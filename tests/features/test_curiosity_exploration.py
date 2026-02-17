import torch
import unittest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.layers.base import MLayer

class TestCuriosityExploration(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.heads = 2
        self.config_on = {
            'active_inference': {
                'curiosity_noise': {
                    'enabled': True,
                    'base_std': 0.5,
                    'sensitivity': 2.0
                }
            }
        }
        self.layer_on = MLayer(self.dim, heads=self.heads, physics_config=self.config_on)
        self.layer_off = MLayer(self.dim, heads=self.heads, physics_config={})
        
    def test_exploration_coverage(self):
        """
        Verifies that curiosity increases the 'volume' of the latent space
        explored by the agent over multiple steps in high-error regions.
        """
        torch.manual_seed(42)
        batch = 10
        steps = 5
        
        # High "Confusion" environment (High Force)
        force = torch.ones(batch, self.dim) * 5.0
        
        x_on = torch.zeros(batch, self.dim)
        v_on = torch.zeros(batch, self.dim)
        
        x_off = torch.zeros(batch, self.dim)
        v_off = torch.zeros(batch, self.dim)
        
        # Track variance across batch (Noise increases variance)
        var_on = 0
        var_off = 0
        
        self.layer_on.train()
        self.layer_off.train()
        
        for _ in range(steps):
            x_on, v_on, _, _ = self.layer_on(x_on, v_on, force=force)
            x_off, v_off, _, _ = self.layer_off(x_off, v_off, force=force)
            
            # Use variance as exploration proxy
            var_on = max(var_on, x_on.var(dim=0).mean().item())
            var_off = max(var_off, x_off.var(dim=0).mean().item())
            
        print(f"\n[Curiosity] Exploration Variance (ON):  {var_on:.6f}")
        print(f"[Curiosity] Exploration Variance (OFF): {var_off:.6f}")
        
        # Curiosity should increase variance significantly
        self.assertGreater(var_on, var_off, 
                          "Curiosity should lead to higher variance in high-confusion zones")

    def test_confusion_correlation(self):
        """
        Verifies that curiosity noise scales with local confusion (force).
        Tests the component in isolation to bypass MLayer's RMSNorm.
        """
        noise_mod = self.layer_on.curiosity_noises[0]
        v = torch.zeros(1000, self.dim // self.heads) # Head dim
        
        # 1. Zero Confusion
        v_zero = noise_mod(v, force=torch.zeros(1000, self.dim // self.heads))
        std_zero = v_zero.std().item()
        
        # 2. High Confusion
        v_high = noise_mod(v, force=torch.ones(1000, self.dim // self.heads) * 5.0)
        std_high = v_high.std().item()
        
        print(f"\n[Curiosity] Component Std (Zero Force): {std_zero:.4f}")
        print(f"[Curiosity] Component Std (High Force): {std_high:.4f}")
        
        self.assertGreater(std_high, std_zero * 2, 
                          "Curiosity noise should scale significantly with force magnitude")

if __name__ == "__main__":
    unittest.main()
