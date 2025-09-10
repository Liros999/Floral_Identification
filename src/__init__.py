"""
Foundational Flower Detector

A high-precision flower detection system using Mask R-CNN with hard negative mining,
optimized for scientific applications and CPU training.

Authors: Based on architectural decisions from project documentation
References:
- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. ICCV.
- Lin, T. Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.
"""

__version__ = "1.0.0"
__author__ = "Foundational Flower Detector Team"
__email__ = "contact@example.com"

# Core modules
from . import config
from . import data_preparation
from . import training
from . import models
from . import verification_ui

__all__ = [
    "config",
    "data_preparation", 
    "training",
    "models",
    "verification_ui"
]
