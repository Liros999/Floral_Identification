# Foundational Flower Detector

A high-precision flower detection system using Mask R-CNN with hard negative mining, optimized for scientific applications and CPU training on Intel Core Ultra 7.

## 🌸 Project Overview

This project implements a foundational flower detection model designed to achieve extremely high precision (≥98%) in distinguishing real flowers from background objects. The system uses an innovative hard negative mining approach to systematically eliminate false positives, making it suitable for scientific applications where precision is critical.

### Key Features

- **High Precision Focus**: Optimized for ≥98% precision with ≥85% recall
- **Hard Negative Mining**: Automated false positive detection and human-in-the-loop correction cycle
- **CPU Optimized**: Designed for Intel Core Ultra 7 with 32GB RAM (no GPU required)
- **Reproducible**: Complete reproducibility with deterministic training and fixed random seeds
- **Scientific Rigor**: Comprehensive testing and validation framework with real data enforcement

## 🏗️ Architecture

The system implements a human-in-the-loop training cycle following the specifications in the project architecture documents:

1. **Initial Training**: Train Mask R-CNN on curated flower vs. background dataset
2. **Automated Mining**: Scan large background datasets for false positives
3. **Human Verification**: Streamlit UI for efficient false positive confirmation
4. **Reinforcement Training**: Retrain model with confirmed hard negatives
5. **Iteration**: Repeat cycle until precision targets are met (≥98% precision, ≥85% recall)

## 🔬 Scientific Foundation

This foundational phase establishes the methodology for the main research proposal by validating that generic object detection can achieve scientific-grade precision through systematic hard negative mining.

### References

- **Mask R-CNN**: He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. ICCV.
- **COCO Dataset**: Lin, T. Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Intel Core Ultra 7 or equivalent (16+ logical cores)
- **RAM**: 32GB (minimum 16GB)
- **Storage**: 100GB free space
- **OS**: Windows 10/11, Linux, or macOS

### Software Requirements
- Python 3.8-3.11
- TensorFlow 2.13.0 (CPU optimized)
- See `requirements.txt` for complete dependency list

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd foundational_flower_detector

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\\Scripts\\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Configuration

The project is pre-configured to use Google Drive data paths. Verify the configuration:

```bash
# Check configuration
python -c "from src.config import Config; c = Config(); print(c.validate())"
```

Edit `config.yaml` if you need to adjust paths or parameters.

### 3. Data Validation

Validate your Google Drive data:

```bash
# Run data validation
flower-detector-build-dataset --validate --verbose
```

### 4. Build Dataset

Create COCO format annotations and train/val/test splits:

```bash
# Build initial dataset
flower-detector-build-dataset --verbose
```

### 5. Training

Start the training process:

```bash
# Initial training
flower-detector-train --config config.yaml --verbose
```

Monitor training progress:
```bash
# Launch TensorBoard
tensorboard --logdir logs/tensorboard --port 6006
```

### 6. Hard Negative Mining

After initial training, find false positives:

```bash
# Mine hard negatives
flower-detector-mine-negatives --model-path models/checkpoints/best_model.h5
```

### 7. Human Verification

Launch the verification UI to confirm false positives:

```bash
# Start verification UI
flower-detector-verify
```

Open your browser to `http://localhost:8501` to begin verification.

### 8. Iterative Improvement

Repeat steps 4-7 until convergence (< 50 new false positives per 50K background images).

## 📊 Project Structure

```
foundational_flower_detector/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── config.py            # Main configuration class
│   ├── data_preparation/         # Data processing utilities
│   │   ├── __init__.py
│   │   ├── utils.py             # Core utilities (reproducibility, validation)
│   │   └── build_dataset.py     # Dataset construction and COCO format
│   ├── training/                 # Model training and mining
│   │   ├── __init__.py
│   │   ├── train.py             # Mask R-CNN training
│   │   └── find_hard_negatives.py  # Automated false positive detection
│   ├── verification_ui/          # Human verification interface
│   │   ├── __init__.py
│   │   └── app.py               # Streamlit verification app
│   └── models/                   # Model architectures
│       ├── __init__.py
│       └── model_config.py      # Mask R-CNN configuration
├── tests/                        # Comprehensive test suite
├── data/                         # Local data storage (gitignored)
├── models/                       # Trained model storage (gitignored)
├── logs/                         # Training logs (gitignored)
├── reports/                      # Generated reports
├── config.yaml                   # Main configuration file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## 📈 Performance Optimization

### CPU Training Optimizations

The system is specifically optimized for Intel Core Ultra 7:

- **Batch Size**: Small batches (2) optimized for 32GB RAM
- **Multi-threading**: Utilizes all 16 logical cores efficiently
- **Intel MKL**: Hardware-accelerated linear algebra operations
- **Memory Management**: Efficient data loading and preprocessing

### Key Configuration Parameters

```yaml
training:
  batch_size: 2                    # Optimized for CPU training
  num_workers: 8                   # Utilize CPU cores efficiently
  learning_rate: 0.001            # Conservative for stability
  epochs: 50                      # More epochs with smaller batches
  gradient_clipping: 1.0          # Stability for CPU training

hardware:
  cpu_cores_logical: 16           # Intel Core Ultra 7
  memory_gb: 32                   # Available system memory
  intraop_threads: 8              # TensorFlow optimization
  enable_mkl: true                # Intel optimization
```

## 📝 Scientific Methodology

### Success Criteria

- **Primary**: Precision ≥ 0.98 on holdout test set
- **Secondary**: Recall ≥ 0.85 on holdout test set
- **Convergence**: <50 new false positives per 50K background images

### Reproducibility

- **Deterministic Training**: Fixed random seeds across all libraries
- **Version Control**: Pinned dependency versions
- **Data Integrity**: Cryptographic hash validation
- **Documentation**: Complete methodology and decision tracking

### Data Sources

- **Positive Images**: Real flower images from validated scientific datasets
- **Negative Images**: COCO dataset background objects (non-plant categories)
- **No Synthetic Data**: Strict enforcement of real data usage only

## 🛠️ Development

### Code Quality

The project follows strict code quality standards:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. Follow the existing architectural patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure reproducibility
5. No synthetic or mock data allowed

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub issues
- **Documentation**: See inline docstrings and architecture documents
- **Configuration**: Refer to `config.yaml` comments for parameter descriptions

## 🏆 Acknowledgments

- **Open Images V7**: Primary positive dataset source
- **COCO Dataset**: Background/negative examples
- **TensorFlow Object Detection API**: Model architecture foundation
- **Intel**: CPU optimization guidance for Ultra 7 architecture

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔬 Citation

If you use this work in your research, please cite:

```bibtex
@software{foundational_flower_detector_2025,
  title={Foundational Flower Detector: High-Precision Flower Detection with Hard Negative Mining},
  author={Foundational Flower Detector Team},
  year={2025},
  url={https://github.com/yourusername/foundational-flower-detector},
  note={Scientific computer vision framework for high-precision object detection}
}
```

---

**Status**: Active Development | **Phase**: Foundational | **Version**: 1.0.0
