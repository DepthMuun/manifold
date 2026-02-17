
import torch


def apply_boundary_python(x, topology_id):
    """
    Apply boundary conditions to position tensor.
    
    IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):
    
    1. TOROIDAL TOPOLOGY:
       For topology_id == 1 (torus), positions are wrapped to [0, 2π).
       The wrapping is periodic: x = x % (2*π)
       
    2. VELOCITY HANDLING:
       Velocity vectors should NOT be wrapped!
       - Position: x is on the manifold, needs wrapping
       - Velocity: v is in the TANGENT SPACE, invariant under wrapping
       
       The wrapping of velocity would create artificial discontinuities
       that break the smoothness of geodesic flow.
       
       If you need to apply velocity corrections, use apply_velocity_correction().
    
    Topology IDs:
    0: Euclidean (None) - No boundary conditions
    1: Toroidal (Periodic [0, 2*PI)) - Positions wrapped
    
    Args:
        x: Position tensor [batch, dim]
        topology_id: Integer topology identifier
        
    Returns:
        Position tensor with boundaries applied
    """
    if topology_id == 1:
        # AUDIT FIX: Use atan2 for smooth, differentiable wrapping
        # This preserves gradient continuity at 0/2π boundaries
        # atan2(sin(x), cos(x)) wraps to [-π, π], shift to [0, 2π)
        PI = 3.14159265359
        TWO_PI = 2.0 * PI
        x_wrapped = torch.atan2(torch.sin(x), torch.cos(x))
        # Shift from [-π, π] to [0, 2π)
        x_wrapped = torch.where(x_wrapped < 0, x_wrapped + TWO_PI, x_wrapped)
        return x_wrapped
    return x


def apply_velocity_correction(v, x_old, x_new, topology_id, dt=1.0):
    """
    Correct velocity for toroidal boundary crossings.
    
    When position crosses the boundary (e.g., from 6.28 to 0.01),
    the apparent velocity is wrong. This function computes the true
    velocity considering boundary crossings.
    
    AUDIT FIX: This function handles velocity correction for torus.
    
    Args:
        v: Velocity tensor [batch, dim]
        x_old: Previous position [batch, dim]
        x_new: Current position [batch, dim]
        topology_id: Topology identifier
        
    Returns:
        Corrected velocity tensor
    """
    if topology_id != 1:
        return v
    
    PI = 3.14159265359
    TWO_PI = 2.0 * PI
    
    # AUDIT FIX: Use atan2 for smooth displacement calculation
    # Compute apparent displacement
    apparent_disp = x_new - x_old
    
    # Compute wrapped displacement with smooth gradients
    wrapped_disp = torch.atan2(torch.sin(apparent_disp), torch.cos(apparent_disp))
    
    if dt == 0:
        dt = 1.0
    return wrapped_disp / dt


def toroidal_dist_python(x1, x2):
    """
    Shortest angular distance on Torus.
    
    IMPORTANT (Auditoria 2026-02-06):
    This computes distance on a FLAT torus (product of circles).
    It does NOT account for the LEARNED Christoffel curvature.
    
    For tasks requiring true geodesic distance on the learned manifold,
    use Christoffel-based distance computation instead.
    
    Args:
        x1: Position tensor [batch, dim]
        x2: Position tensor [batch, dim]
        
    Returns:
        Distance tensor
    """
    PI = 3.14159265359
    diff = x1 - x2
    # Wrap difference to [-pi, pi]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.norm(diff, dim=-1)


def resolve_topology_id(christoffel, topology_id_arg=None):
    """
    Resolve topology ID from Christoffel geometry or argument.
    
    Args:
        christoffel: The Christoffel geometry object
        topology_id_arg: Optional override from kwargs
        
    Returns:
        Integer topology ID (0=Euclidean, 1=Torus)
    """
    # 1. Use argument if provided
    if topology_id_arg is not None:
        return topology_id_arg
        
    # 2. Check Christoffel attribute
    tid = getattr(christoffel, 'topology_id', 0)
    
    # 3. Check legacy boolean flag
    if tid == 0 and hasattr(christoffel, 'is_torus') and christoffel.is_torus:
        return 1
        
    return tid


def get_boundary_features(x, topology_id):
    """
    Extract features relevant to the topology boundary.
    
    For Euclidean (0): Returns x
    For Toroidal (1): Returns [sin(x), cos(x)]
    
    Args:
        x: Position tensor [batch, dim]
        topology_id: Integer topology identifier
        
    Returns:
        Feature tensor [batch, dim] or [batch, 2*dim]
    """
    if topology_id == 1:
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    return x
