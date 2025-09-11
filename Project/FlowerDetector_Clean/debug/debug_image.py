"""Debug image loading to see what's causing validation failures."""

from PIL import Image
from pathlib import Path

def debug_image_validation(image_path):
    """Debug why image validation is failing."""
    print(f"🔍 Debugging image: {Path(image_path).name}")
    
    try:
        # Try to open image
        with Image.open(image_path) as img:
            print(f"✅ Image opened successfully")
            print(f"   - Size: {img.size}")
            print(f"   - Mode: {img.mode}")
            print(f"   - Format: {img.format}")
            
            # Check dimensions
            width, height = img.size
            if width < 32 or height < 32:
                print(f"❌ Image too small: {width}x{height}")
                return False
            else:
                print(f"✅ Size OK: {width}x{height}")
            
            # Check mode
            if img.mode not in ['RGB', 'RGBA', 'L']:
                print(f"❌ Unsupported mode: {img.mode}")
                return False
            else:
                print(f"✅ Mode OK: {img.mode}")
            
            # Try to convert to RGB
            img_rgb = img.convert('RGB')
            print(f"✅ RGB conversion successful")
            
            # The problematic part - verify()
            print("🔍 Calling img.verify()...")
            
            # We need to reopen because verify() can corrupt the image object
            with Image.open(image_path) as img2:
                img2.verify()
            
            print(f"✅ Verification successful")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

# Test a few images
test_images = [
    r"G:\My Drive\Floral_Detector\Phase1_Foundational-Detector\Phase1_Data\raw_data\positive_images\0000626c3e317247.jpg",
    r"G:\My Drive\Floral_Detector\Phase1_Foundational-Detector\Phase1_Data\raw_data\negative_images (1)\negative_0009_97165.jpg"
]

for img_path in test_images:
    print("=" * 60)
    debug_image_validation(img_path)
    print()
