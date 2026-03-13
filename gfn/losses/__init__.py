"""
gfn/losses/__init__.py
Public API del módulo losses — GFN V5
"""

from gfn.losses.base import BaseLoss
from gfn.losses.factory import LossFactory
from gfn.losses.generative import ManifoldGenerativeLoss
from gfn.losses.physics import PhysicsLoss, PhysicsInformedLoss
from gfn.losses.toroidal import ToroidalLoss, ToroidalVelocityLoss, ToroidalDistanceLoss
from gfn.losses.regularization import NoetherSymmetryLoss, DynamicLossBalancer
from gfn.losses.detection import GIoULoss, IoULoss, giou_loss, iou_loss

__all__ = [
    "BaseLoss",
    "LossFactory",
    "ManifoldGenerativeLoss",
    "PhysicsLoss",
    "PhysicsInformedLoss",
    "ToroidalLoss",
    "ToroidalVelocityLoss",
    "ToroidalDistanceLoss",
    "NoetherSymmetryLoss",
    "DynamicLossBalancer",
    # Detection
    "GIoULoss", "IoULoss", "giou_loss", "iou_loss",
]
