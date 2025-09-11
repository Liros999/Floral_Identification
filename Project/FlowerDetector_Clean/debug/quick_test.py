"""Quick test of PyTorch modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

print("🚀 Quick PyTorch Module Test")
print("=" * 40)

# Test 1: Configuration
try:
    from data_preparation.config_loader import ConfigLoader
    config_loader = ConfigLoader()
    config = config_loader.load()
    print("✅ Config loaded successfully")
    print(f"   Target precision: {config_loader.get('training.target_precision')}")
except Exception as e:
    print(f"❌ Config failed: {e}")

# Test 2: Simple model creation
try:
    import torch
    from model.flower_classifier import create_flower_classifier
    
    model = create_flower_classifier(config)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print("✅ PyTorch model works")
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Output values: {output.squeeze().tolist()}")
    
except Exception as e:
    print(f"❌ Model failed: {e}")

print("\n🎉 Quick test complete!")
