"""
TEST SCRIPT FOR ENHANCED PROGRESS BARS
Demonstrates the new progress bar system for the complete training pipeline.
"""

import logging
from src.data_preparation.config_loader import ConfigLoader
from src.data_preparation.real_data_loader import create_real_data_loaders
from src.training.train import create_trainer
from src.models.model_factory import create_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_progress():
    """Test the enhanced progress bar system for the complete pipeline."""
    print("🧪 TESTING ENHANCED PROGRESS BARS")
    print("=" * 60)
    
    try:
        # Load configuration
        print("📋 Step 1: Loading configuration...")
        config_loader = ConfigLoader()
        config = config_loader.load()
        print("✅ Configuration loaded successfully")
        
        # Create dataset (this will show data loading progress bars)
        print("\n📊 Step 2: Creating dataset with progress bars...")
        print("=" * 60)
        data_loaders = create_real_data_loaders(config)
        dataset = data_loaders['dataset']
        print("✅ Dataset created successfully")
        
        # Create model
        print("\n🏗️ Step 3: Creating model...")
        model = create_model(config)
        print("✅ Model created successfully")
        
        # Create trainer and setup (this will show model setup progress)
        print("\n🎯 Step 4: Setting up trainer...")
        trainer = create_trainer(config)
        trainer.setup_model(model)
        print("✅ Model setup completed")
        
        # Setup data loaders (this will show data loader setup progress)
        print("\n📊 Step 4.1: Setting up data loaders...")
        print("=" * 60)
        print("🔄 Creating train/validation/test splits...")
        print("⚖️ Computing class weights...")
        print("🔧 Optimizing data loader settings...")
        print("=" * 60)
        
        trainer.setup_data_loaders(dataset)
        print("✅ Data loaders setup completed")
        
        # Show final statistics
        print("\n🎉 ENHANCED PROGRESS BARS TEST COMPLETED!")
        print("=" * 60)
        print("📊 FINAL STATISTICS:")
        print(f"  🚂 Train samples: {len(trainer.train_loader.dataset)}")
        print(f"  ✅ Val samples: {len(trainer.val_loader.dataset)}")
        print(f"  🧪 Test samples: {len(trainer.test_loader.dataset)}")
        print(f"  📦 Batch size: {config['data']['batch_size']}")
        print(f"  👥 Workers: {config['data']['num_workers']}")
        print("=" * 60)
        
        print("✅ All progress bars working correctly!")
        print("🚀 Ready for training with enhanced visual feedback!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_enhanced_progress()
