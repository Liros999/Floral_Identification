"""
Setup script for Foundational Flower Detector.

This setup script configures the project for installation and distribution,
following Python packaging best practices and scientific software standards.

References:
- Python Packaging Authority guidelines
- Scientific Python development best practices
- TensorFlow ecosystem compatibility

Author: Foundational Flower Detector Team
Date: September 2025
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read README for long description
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "Foundational Flower Detector - High-precision flower detection with hard negative mining"

# Read requirements
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as fh:
            return [
                line.strip() 
                for line in fh 
                if line.strip() and not line.startswith("#") and not line.startswith("--")
            ]
    return []

# Read version from package
def get_version():
    """Extract version from package."""
    try:
        from src import __version__
        return __version__
    except ImportError:
        return "1.0.0"

setup(
    name="foundational-flower-detector",
    version=get_version(),
    author="Foundational Flower Detector Team",
    author_email="contact@example.com",
    description="High-precision flower detection with hard negative mining for scientific applications",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/foundational-flower-detector",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/foundational-flower-detector/issues",
        "Source": "https://github.com/yourusername/foundational-flower-detector",
        "Documentation": "https://foundational-flower-detector.readthedocs.io/",
    },
    
    # Package configuration
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    
    # Dependencies
    python_requires=">=3.8",  # TensorFlow 2.13 compatibility
    install_requires=read_requirements(),
    
    # Optional dependencies for different use cases
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.1",
            "pytest-mock>=3.11.1",
            "pytest-timeout>=2.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.1",
        ],
        "docs": [
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "tensorflow-gpu==2.13.0",
        ],
        "intel": [
            # Intel optimizations (may need separate installation)
            # "mkl>=2023.2.0",
            # "intel-openmp>=2023.2.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.5.4",
            "ipywidgets>=8.1.0",
            "jupyterlab>=3.6.5",
        ],
        "monitoring": [
            "wandb>=0.15.8",
            "tensorboard>=2.13.0",
            "memory-profiler>=0.61.0",
            "line-profiler>=4.1.1",
        ],
        "full": [
            # All optional dependencies
            "pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.7.0",
            "sphinx>=7.1.2", "wandb>=0.15.8", "jupyter>=1.0.0",
        ]
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "flower-detector-build-dataset=src.data_preparation.build_dataset:main",
            "flower-detector-train=src.training.train:main",
            "flower-detector-mine-negatives=src.training.find_hard_negatives:main",
            "flower-detector-verify=src.verification_ui.app:main",
            "flower-detector-validate=src.data_preparation.utils:main",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: TensorFlow",
    ],
    
    # Keywords for searchability
    keywords=[
        "computer-vision", "object-detection", "mask-rcnn", "flowers", 
        "scientific-computing", "machine-learning", "hard-negative-mining",
        "tensorflow", "cpu-optimization", "reproducible-science"
    ],
    
    # Package data
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt", "*.md"],
        "src": ["config/*.yaml"],
    },
    
    # Data files
    data_files=[
        ("config", ["config.yaml"]),
    ],
    
    # Zip safety
    zip_safe=False,
    
    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.1",
    ],
    
    # License
    license="MIT",
    
    # Platform requirements
    platforms=["any"],
)
