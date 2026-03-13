import gfn.contracts as contracts
from typing import Optional, Tuple, Any

class MockGeometry:
    def forward(self, x: Any, v: Optional[Any] = None, **kwargs: Any) -> Any:
        return x
    def metric_tensor(self, x: Any) -> Any:
        return x

class MockIntegrator:
    def forward(self, x: Any, v: Any, force: Optional[Any] = None, dt: float = 0.1, **kwargs: Any) -> Tuple[Any, Any, Any]:
        return x, v, None

def test_protocols():
    geo = MockGeometry()
    print(f"MockGeometry implements Geometry: {isinstance(geo, contracts.Geometry)}")
    
    integ = MockIntegrator()
    print(f"MockIntegrator implements Integrator: {isinstance(integ, contracts.Integrator)}")

    # Negative test
    class BrokenGeometry:
        def forward(self, x): pass
    
    broken = BrokenGeometry()
    print(f"BrokenGeometry implements Geometry: {isinstance(broken, contracts.Geometry)}")

if __name__ == "__main__":
    test_protocols()
