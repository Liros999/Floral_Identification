# Foundational Flower Detector - Usage Guide

## 🚀 Quick Start Guide

This guide will help you get the Foundational Flower Detector up and running with your Google Drive data.

### Prerequisites

- Python 3.8-3.11
- Intel Core Ultra 7 (or similar CPU)
- 32GB RAM (minimum 16GB)
- Access to your Google Drive data at: `G:\My Drive\Floral_Detector\Phase1_Foundational-Detector\Phase1_Data`

### 1. Environment Setup

```bash
# Navigate to project directory
cd "C:\Users\lirdi\Desktop\Projects\Floral_Identification\Project\foundational_flower_detector"

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### 2. Verify Installation

```bash
# Test configuration loading
python -c "from src.config import Config; c = Config(); print('✅ Config loaded successfully')"

# Validate system requirements
python -c "from src.data_preparation.utils import SystemMonitor; print('✅ System check:', SystemMonitor.check_system_requirements({'hardware': {'memory_gb': 16, 'cpu_cores_logical': 8}})['system_ready'])"
```

### 3. Data Validation

```bash
# Validate your Google Drive data
flower-detector-build-dataset --validate --verbose
```

**Expected Output:**
```
INFO - positive_images: 99.5% valid (1061/1066)
INFO - Found background images for training
INFO - Configuration validation passed
```

### 4. Build Initial Dataset

```bash
# Create COCO format annotations and train/val/test splits
flower-detector-build-dataset --verbose
```

**Expected Output:**
```
INFO - Loaded 1061 positive flower images
INFO - Loaded 1995 negative background images  
INFO - Created train/val/test splits deterministically
INFO - Generated COCO JSON annotations (version 1)
INFO - Dataset building completed successfully!
```

### 5. Start Training

```bash
# Begin Mask R-CNN training
flower-detector-train --config config.yaml --verbose
```

**Expected Output:**
```
INFO - Building Mask R-CNN model
INFO - Applied CPU optimizations for Intel Core Ultra 7
INFO - Training for 50 epochs with batch size 2
INFO - System requirements met: memory=32GB, cores=16
Epoch 1/50
████████████████████████████████ 150/150 [02:30<00:00, 1.00it/s] - loss: 2.1234 - precision: 0.7234
```

**Training will take approximately 4-6 hours on Intel Core Ultra 7**

### 6. Monitor Training Progress

In a new terminal:
```bash
# Launch TensorBoard
tensorboard --logdir logs/tensorboard --port 6006
```

Open your browser to: `http://localhost:6006`

### 7. Hard Negative Mining

After initial training completes:

```bash
# Mine hard negatives (false positives)
flower-detector-mine-negatives --verbose
```

**Expected Output:**
```
INFO - Loading best model for hard negative mining
INFO - Scanning 50000 background images for false positives
INFO - Found 234 false positives (confidence > 0.90)
INFO - Verification queue created: verification_queue.json
```

### 8. Human Verification

```bash
# Launch verification UI
flower-detector-verify
```

**Opens Streamlit app at:** `http://localhost:8501`

**In the UI:**
1. Review each detected false positive
2. Click "✅ Confirm FP" if it's truly NOT a flower
3. Click "❌ Reject FP" if it IS actually a flower
4. Work through the queue systematically

### 9. Iterative Improvement

After confirming hard negatives:

```bash
# Rebuild dataset with confirmed hard negatives
flower-detector-build-dataset --verbose

# Retrain model with augmented negative examples
flower-detector-train --config config.yaml --verbose

# Repeat mining cycle
flower-detector-mine-negatives --verbose
```

**Repeat until convergence** (< 50 new false positives per 50K background images)

## 📊 Expected Performance Timeline

| Stage | Duration | Expected Results |
|-------|----------|------------------|
| Initial Training | 4-6 hours | Precision: ~85-90%, Recall: ~80-85% |
| First Hard Negative Mining | 2-3 hours | Find: 200-500 false positives |
| Human Verification | 1-2 hours | Confirm: ~60-80% of candidates |
| Second Training Cycle | 4-6 hours | Precision: ~92-95%, Recall: ~82-87% |
| Second Mining Cycle | 2-3 hours | Find: 100-200 false positives |
| Convergence (3-4 cycles) | 2-3 days total | **Precision: ≥98%, Recall: ≥85%** |

## 🎯 Success Criteria

The project achieves success when:

- ✅ **Precision ≥ 0.98** on holdout test set
- ✅ **Recall ≥ 0.85** on holdout test set  
- ✅ **< 50 new false positives** per 50K background images

## 🔧 Configuration Customization

Edit `config.yaml` to customize:

```yaml
# Adjust for your hardware
training:
  batch_size: 4        # Increase if you have more RAM
  num_workers: 12      # Increase for more CPU cores
  epochs: 100          # Increase for better convergence

# Adjust mining sensitivity
hard_negative_mining:
  confidence_threshold: 0.95    # Higher = fewer but harder negatives
  scan_limit: 100000           # More images to scan
```

## 📁 Understanding the Output Files

After successful training, you'll have:

```
foundational_flower_detector/
├── models/checkpoints/
│   ├── best_model.h5           # Best performing model weights
│   └── final_model.h5          # Final training epoch weights
├── logs/
│   ├── training.log            # Detailed training logs
│   ├── tensorboard/            # TensorBoard visualization data
│   └── training_metrics.jsonl # JSON logs for analysis
├── reports/
│   └── training_results_*.json # Complete training results
├── data/processed/annotations/
│   ├── train_annotations_v*.json  # COCO format training data
│   ├── val_annotations_v*.json    # COCO format validation data
│   └── test_annotations_v*.json   # COCO format test data
└── verification_queue.json    # Current hard negatives to review
```

## 🚨 Troubleshooting

### Common Issues

**1. "No model weights found"**
```bash
# Check if training completed
ls models/checkpoints/
# Should see best_model.h5 or final_model.h5
```

**2. "Memory allocation failed"**
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

**3. "No positive images found"**
```bash
# Verify Google Drive path
ls "G:\My Drive\Floral_Detector\Phase1_Foundational-Detector\Phase1_Data\raw_data\positive_images"
# Should show ~1000 flower images
```

**4. "TensorFlow GPU not found" warnings**
```
# This is expected - we're using CPU-only training
# These warnings can be safely ignored
```

### Performance Optimization

**For faster training:**
```yaml
# In config.yaml
training:
  batch_size: 4              # If you have >32GB RAM
  num_workers: 16            # Use all logical cores
  validation_frequency: 5    # Validate every 5 epochs instead of 1
```

**For better precision:**
```yaml
# In config.yaml
hard_negative_mining:
  confidence_threshold: 0.95  # Higher threshold = harder negatives
  max_cycles: 15             # More mining cycles
```

## 📞 Getting Help

1. **Check logs:** `tail -f logs/training.log`
2. **System status:** `flower-detector-build-dataset --validate`
3. **Resource monitoring:** Check `logs/training_metrics.jsonl`

## ✅ Verification Checklist

Before proceeding to the next phase, ensure:

- [ ] Training completed without errors
- [ ] Test precision ≥ 98%
- [ ] Test recall ≥ 85%
- [ ] Hard negative mining converged (< 50 FPs per 50K images)
- [ ] All verification queues processed
- [ ] Model weights saved in `models/checkpoints/`
- [ ] Training results documented in `reports/`

**🎉 Congratulations!** You now have a scientifically validated foundational flower detector ready for the next research phase.
