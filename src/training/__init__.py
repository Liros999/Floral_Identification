"""
Training pipeline for Foundational Flower Detector.

Implements the hard negative mining cycle with:
- Mask R-CNN training with CPU optimization
- Automated false positive detection
- Model performance tracking
- Scientific evaluation metrics
"""

from .train import MaskRCNNTrainer
from .find_hard_negatives import HardNegativeMiner

__all__ = [
    "MaskRCNNTrainer",
    "HardNegativeMiner"
]
