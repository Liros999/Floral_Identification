"""
Model architecture definitions for Foundational Flower Detector.

Implements Mask R-CNN based architecture following:
- He et al. (2017) Mask R-CNN methodology
- CPU-optimized configurations
- Scientific evaluation standards
"""

from .model_config import MaskRCNNConfig, ModelFactory

__all__ = [
    "MaskRCNNConfig",
    "ModelFactory"
]
