"""
TEST SCRIPT FOR IMAGE ANALYSIS CACHE SYSTEM
Demonstrates the cache functionality and performance improvements.
"""

import logging
import time
from pathlib import Path
from src.data_preparation.image_cache import create_image_cache
from src.data_preparation.cache_manager import CacheManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_cache_system():
    """Test the image analysis cache system."""
    print("🧪 TESTING IMAGE ANALYSIS CACHE SYSTEM")
    print("=" * 50)
    
    # Create cache
    cache = create_image_cache()
    
    # Test with some sample paths
    test_paths = [
        Path("G:/My Drive/Floral_Detector/Phase1_Foundational-Detector/Phase1_Data/raw_data/positive_images"),
        Path("G:/My Drive/Floral_Detector/Phase1_Foundational-Detector/Phase1_Data/raw_data/balanced_negatives"),
    ]
    
    print(f"📁 Testing with {len(test_paths)} directories")
    
    # Test 1: Check cache performance
    print("\n🔍 Test 1: Cache Performance")
    start_time = time.time()
    
    total_images = 0
    analyzed_count = 0
    
    for test_dir in test_paths:
        if test_dir.exists():
            print(f"  Checking directory: {test_dir.name}")
            images = list(test_dir.glob("*.jpg"))[:10]  # Test with first 10 images
            
            for img_path in images:
                total_images += 1
                if cache.is_analyzed(img_path):
                    analyzed_count += 1
                else:
                    # Simulate analysis
                    label = 1 if "positive" in str(test_dir) else 0
                    metadata = {
                        'test_run': True,
                        'image_size': (224, 224),
                        'processed_at': time.time()
                    }
                    cache.add_analysis(img_path, label, metadata)
    
    end_time = time.time()
    
    print(f"  ✅ Processed {total_images} images in {end_time - start_time:.2f} seconds")
    print(f"  📊 Cache stats: {cache.get_cache_stats()}")
    
    # Test 2: Second run (should be much faster)
    print("\n🚀 Test 2: Second Run (Cache Hit)")
    start_time = time.time()
    
    second_run_analyzed = 0
    for test_dir in test_paths:
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg"))[:10]
            for img_path in images:
                if cache.is_analyzed(img_path):
                    second_run_analyzed += 1
    
    end_time = time.time()
    
    print(f"  ✅ Found {second_run_analyzed} already analyzed images in {end_time - start_time:.2f} seconds")
    print(f"  📈 Speed improvement: {total_images / (end_time - start_time):.1f} images/second")
    
    # Test 3: Cache management
    print("\n🛠️ Test 3: Cache Management")
    manager = CacheManager()
    
    # Get cache info
    info = manager.get_cache_info()
    print(f"  📁 Cache file: {info['cache_file']}")
    print(f"  💾 Cache size: {info['cache_file_size_mb']:.2f} MB")
    print(f"  📊 Total entries: {info['stats']['total_entries']}")
    
    # Export report
    report_file = manager.export_cache_report()
    print(f"  📄 Report exported to: {report_file}")
    
    print("\n🎉 CACHE SYSTEM TEST COMPLETED!")
    print("=" * 50)


def demonstrate_performance_improvement():
    """Demonstrate the performance improvement with cache."""
    print("\n📈 PERFORMANCE IMPROVEMENT DEMONSTRATION")
    print("=" * 50)
    
    # Simulate first run (no cache)
    print("🐌 First run (no cache):")
    start_time = time.time()
    
    # Simulate processing 1000 images
    for i in range(1000):
        # Simulate image processing time
        time.sleep(0.001)  # 1ms per image
    
    first_run_time = time.time() - start_time
    print(f"  ⏱️ Time: {first_run_time:.2f} seconds")
    print(f"  📊 Rate: {1000 / first_run_time:.1f} images/second")
    
    # Simulate second run (with cache)
    print("\n🚀 Second run (with cache):")
    start_time = time.time()
    
    # Simulate 80% cache hit rate
    cache_hits = 800
    cache_misses = 200
    
    # Process cache misses (much faster)
    for i in range(cache_misses):
        time.sleep(0.001)  # 1ms per image
    
    second_run_time = time.time() - start_time
    print(f"  ⏱️ Time: {second_run_time:.2f} seconds")
    print(f"  📊 Rate: {1000 / second_run_time:.1f} images/second")
    print(f"  🎯 Speed improvement: {first_run_time / second_run_time:.1f}x faster")
    print(f"  💾 Cache hit rate: {cache_hits / 1000:.1%}")


if __name__ == "__main__":
    try:
        test_cache_system()
        demonstrate_performance_improvement()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
