"""
TEST SCRIPT FOR ENHANCED PROGRESS BARS
Demonstrates the new progress bar system for data loading.
"""

import logging
from src.data_preparation.config_loader import ConfigLoader
from src.data_preparation.real_data_loader import create_real_data_loaders

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_progress_bars():
    """Test the enhanced progress bar system."""
    print("🧪 TESTING ENHANCED PROGRESS BARS")
    print("=" * 60)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config_loader = ConfigLoader()
        config = config_loader.load()
        
        print("✅ Configuration loaded successfully")
        print(f"📁 Base path: {config['data_paths']['base']}")
        print(f"🌺 Positive images: {config['data_paths']['positive_images']}")
        print(f"🌿 Negative images: {config['data_paths']['negative_images']}")
        
        print("\n🚀 Starting data loading with progress bars...")
        print("=" * 60)
        
        # Create data loaders (this will show the progress bars)
        data_loaders = create_real_data_loaders(config)
        
        print("\n🎉 Data loading completed successfully!")
        print("=" * 60)
        
        # Show final statistics
        dataset = data_loaders['dataset']
        distribution = data_loaders['distribution']
        
        print("📊 FINAL STATISTICS:")
        print(f"  🌺 Positive images: {distribution['positive_flowers']}")
        print(f"  🌿 Negative images: {distribution['negative_background']}")
        print(f"  📈 Total images: {distribution['total']}")
        print(f"  ⚖️ Balance ratio: {distribution['balance_ratio']:.2f}")
        
        # Show cache statistics if available
        if hasattr(dataset, 'get_analysis_cache_stats'):
            cache_stats = dataset.get_analysis_cache_stats()
            if cache_stats.get('enabled', False):
                print(f"\n💾 CACHE STATISTICS:")
                print(f"  📊 Total entries: {cache_stats['total_entries']}")
                print(f"  📈 Hit rate: {cache_stats['hit_rate']:.1%}")
                print(f"  ⚡ Cache hits: {cache_stats['hits']}")
                print(f"  🔄 Cache misses: {cache_stats['misses']}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_progress_bars()
