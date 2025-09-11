"""
Quick TensorBoard test - 1 epoch only to populate dashboard.
Tests all advanced logging features without waiting for full training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from src.data_preparation.config_loader import ConfigLoader
from src.data_preparation.real_data_loader import FlowerDataset, create_real_data_transforms
from src.model.flower_classifier import create_flower_classifier
from src.training.train import create_trainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tensorboard_quick():
    """Run 1 epoch to populate TensorBoard dashboard quickly."""
    
    print("🚀 QUICK TENSORBOARD TEST - 1 EPOCH ONLY")
    print("=" * 50)
    print("Testing: Graph, Images, Histograms, Scalars")
    print("=" * 50)
    
    try:
        # Load config (explicit path relative to project root)
        project_root = Path(__file__).resolve().parents[1]
        config_loader = ConfigLoader(config_path=str(project_root / "config.yaml"))
        config = config_loader.load()
        
        # Override for quick test
        config['training']['epochs'] = 1
        config['training']['quick_test_mode'] = True
        config['training']['quick_test_images_per_class'] = 10  # Very small dataset
        
        print(f"✅ Config loaded - 1 epoch, 10 images per class")
        
        # Create minimal dataset
        data_paths = config['data_paths']
        transforms_dict = create_real_data_transforms()
        
        dataset = FlowerDataset(
            positive_dir=data_paths['positive_images'],
            negative_dir=data_paths['negative_images'],
            transform=transforms_dict['train'],
            max_images_per_class=10,  # Very small for speed
            validate_images=True
        )
        
        distribution = dataset.get_class_distribution()
        print(f"✅ Dataset created: {distribution['total']} images")
        
        # Create model
        model = create_flower_classifier(config)
        print(f"✅ Model created")
        
        # Create trainer
        trainer = create_trainer(config)
        trainer.setup_model(model)
        trainer.setup_data_loaders(dataset)
        
        print(f"✅ Trainer setup complete")
        print(f"   - Train samples: {len(trainer.train_loader.dataset)}")
        print(f"   - Val samples: {len(trainer.val_loader.dataset)}")
        
        # Run 1 epoch
        print(f"\n🔥 RUNNING 1 EPOCH FOR TENSORBOARD...")
        print(f"📊 Watch for: Scalars, Images, Histograms, Graph")
        print("-" * 50)
        
        results = trainer.train()
        
        print(f"\n✅ QUICK TEST COMPLETED!")
        print(f"   - Epochs: {results['epochs_trained']}")
        print(f"   - Final precision: {results['final_metrics']['precision']:.3f}")
        print(f"   - Final accuracy: {results['final_metrics']['accuracy']:.3f}")
        print(f"\n🌐 TensorBoard Dashboard:")
        # Use actual chosen port instead of hardcoded 6006
        actual_port = getattr(trainer, '_tb_port', 6006)
        print(f"   - URL: http://localhost:{actual_port}")
        print(f"   - Log dir: {trainer.log_dir}")
        print(f"   - Check: Scalars, Images, Histograms, Graphs tabs")
        
        return True
        
    except Exception as e:
        print(f"\n❌ QUICK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tensorboard_quick()
    sys.exit(0 if success else 1)
