"""
REAL DATA TEST - NO DUMMY DATA ALLOWED
Tests our PyTorch model with ACTUAL Google Drive flower images.
Ensures model integrity with real scientific data only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_real_data_pipeline():
    """Test complete pipeline with REAL Google Drive images."""
    
    print("🌸 REAL FLOWER DATA TEST")
    print("=" * 50)
    print("CRITICAL: NO DUMMY DATA - ONLY REAL IMAGES")
    print("=" * 50)
    
    try:
        # 1. Load configuration
        from data_preparation.config_loader import ConfigLoader
        
        config_loader = ConfigLoader()
        config = config_loader.load()
        
        print("✅ Configuration loaded")
        print(f"   - Positive images: {config['data_paths']['positive_images']}")
        print(f"   - Negative images: {config['data_paths']['negative_images']}")
        
        # 2. Test REAL data loading
        from data_preparation.real_data_loader import test_real_data_loading
        
        real_data_success = test_real_data_loading(config)
        if not real_data_success:
            print("❌ REAL data loading failed!")
            return False
        
        # 3. Create model and test with REAL data
        from model.flower_classifier import create_flower_classifier
        from data_preparation.real_data_loader import create_real_data_loaders
        
        print("\n🧠 Testing model with REAL flower images...")
        
        # Create model
        model = create_flower_classifier(config)
        model.eval()
        
        # Load REAL data
        data_loaders = create_real_data_loaders(config)
        train_loader = data_loaders['train']
        
        # Test inference on REAL images
        real_images, real_labels = next(iter(train_loader))
        
        print(f"📊 REAL DATA BATCH:")
        print(f"   - Image batch shape: {real_images.shape}")
        print(f"   - Real labels: {real_labels.tolist()}")
        
        # Model inference on REAL data
        with torch.no_grad():
            logits = model(real_images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        print(f"🔮 MODEL PREDICTIONS ON REAL IMAGES:")
        print(f"   - Logits shape: {logits.shape}")
        print(f"   - Sample probabilities: {probabilities[:3].tolist()}")
        print(f"   - Predictions: {predictions.tolist()}")
        print(f"   - Actual labels: {real_labels.tolist()}")
        
        # Validate predictions are reasonable
        if logits.shape != (real_images.shape[0], 2):
            raise ValueError(f"Wrong output shape: {logits.shape}")
        
        if not torch.all((predictions >= 0) & (predictions <= 1)):
            raise ValueError(f"Invalid predictions: {predictions}")
        
        # Check class distribution
        distribution = data_loaders['distribution']
        print(f"\n📈 REAL DATA STATISTICS:")
        print(f"   - Total images: {distribution['total']}")
        print(f"   - Positive flowers: {distribution['positive_flowers']}")
        print(f"   - Negative backgrounds: {distribution['negative_background']}")
        print(f"   - Balance ratio: {distribution['balance_ratio']:.2f}")
        
        print("\n🎉 SUCCESS: Model working with REAL flower data!")
        print("✅ NO dummy data detected")
        print("✅ REAL Google Drive images loaded")
        print("✅ Model produces valid predictions")
        print("✅ Data integrity maintained")
        
        return True
        
    except Exception as e:
        print(f"\n❌ REAL DATA TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run REAL data test."""
    success = test_real_data_pipeline()
    
    if success:
        print("\n🌟 REAL DATA PIPELINE VALIDATED!")
        print("Ready to proceed with training on actual flower images.")
    else:
        print("\n⚠️  REAL DATA TEST FAILED!")
        print("Must fix issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
