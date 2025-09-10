"""
Configuration management for Foundational Flower Detector.

Provides centralized configuration handling with support for:
- CPU-optimized training parameters
- Google Drive data path management
- Reproducibility settings
- Hardware optimization for Intel Core Ultra 7
"""

from .config import Config

__all__ = ["Config"]
