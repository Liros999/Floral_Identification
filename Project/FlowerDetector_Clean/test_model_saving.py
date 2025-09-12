"""
TEST SCRIPT FOR MODEL SAVING AND LOADING
Verifies that model weights are saved and can be loaded correctly.
"""

import logging
from pathlib import Path
from src.data_preparation.config_loader import ConfigLoader
from src.training.train import create_trainer
from src.model.flower_classifier import create_flower_classifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_saving():
    """Test model saving and loading functionality."""
    print("🧪 TESTING MODEL SAVING AND LOADING")
    print("=" * 60)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config_loader = ConfigLoader()
        config = config_loader.load()
        print("✅ Configuration loaded")
        
        # Create model
        print("🏗️ Creating model...")
        model = create_flower_classifier(config)
        print("✅ Model created")
        
        # Create trainer
        print("🎯 Creating trainer...")
        trainer = create_trainer(config)
        trainer.setup_model(model)
        print("✅ Trainer setup complete")
        
        # Test model saving
        print("\n💾 Testing model saving...")
        test_model_path = Path("models/checkpoints/test_save_load.pth")
        trainer.save_model(test_model_path)
        print(f"✅ Model saved to: {test_model_path}")
        
        # Verify file exists and has reasonable size
        if test_model_path.exists():
            file_size = test_model_path.stat().st_size
            print(f"📊 Model file size: {file_size / (1024*1024):.1f} MB")
            if file_size > 10 * 1024 * 1024:  # At least 10MB
                print("✅ Model file size looks reasonable")
            else:
                print("⚠️ Model file seems small, might be incomplete")
        else:
            print("❌ Model file was not created!")
            return False
        
        # Test model loading
        print("\n🔄 Testing model loading...")
        try:
            # Create a new trainer and model for loading test
            new_trainer = create_trainer(config)
            new_model = create_flower_classifier(config)
            new_trainer.setup_model(new_model)
            
            # Load the saved model
            new_trainer.load_model(test_model_path)
            print("✅ Model loaded successfully")
            
            # Verify the loaded model has the same structure
            original_params = sum(p.numel() for p in model.parameters())
            loaded_params = sum(p.numel() for p in new_trainer.model.parameters())
            
            if original_params == loaded_params:
                print(f"✅ Parameter count matches: {original_params:,} parameters")
            else:
                print(f"❌ Parameter count mismatch: {original_params:,} vs {loaded_params:,}")
                return False
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
        
        # Test inference with loaded model
        print("\n🔮 Testing inference with loaded model...")
        try:
            import torch
            # Create a dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Test original model
            model.eval()
            with torch.no_grad():
                original_output = model(dummy_input)
            
            # Test loaded model
            new_trainer.model.eval()
            with torch.no_grad():
                loaded_output = new_trainer.model(dummy_input)
            
            # Compare outputs
            if torch.allclose(original_output, loaded_output, atol=1e-6):
                print("✅ Inference outputs match between original and loaded models")
            else:
                print("❌ Inference outputs don't match")
                return False
                
        except Exception as e:
            print(f"❌ Inference test failed: {e}")
            return False
        
        print("\n🎉 ALL MODEL SAVING TESTS PASSED!")
        print("=" * 60)
        print("✅ Model saving works correctly")
        print("✅ Model loading works correctly")
        print("✅ Parameter counts match")
        print("✅ Inference outputs match")
        print("✅ Model weights are properly preserved")
        
        # Clean up test file
        try:
            test_model_path.unlink()
            print(f"🧹 Cleaned up test file: {test_model_path}")
        except Exception as e:
            print(f"⚠️ Could not clean up test file: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_model_saving()
    if success:
        print("\n🚀 Model saving system is working perfectly!")
    else:
        print("\n💥 Model saving system has issues that need fixing!")
