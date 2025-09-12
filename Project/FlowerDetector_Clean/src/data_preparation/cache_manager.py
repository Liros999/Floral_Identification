"""
CACHE MANAGEMENT UTILITY
Provides tools to manage the image analysis cache system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

from .image_cache import ImageAnalysisCache, create_image_cache

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Utility class for managing image analysis cache.
    Provides high-level operations for cache maintenance.
    """
    
    def __init__(self, cache_dir: Path = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory containing cache files
        """
        self.cache = create_image_cache(cache_dir)
        self.cache_dir = self.cache.cache_file.parent
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        stats = self.cache.get_cache_stats()
        
        # Add file system info
        cache_file_size = self.cache.cache_file.stat().st_size if self.cache.cache_file.exists() else 0
        
        return {
            'cache_file': str(self.cache.cache_file),
            'cache_file_size_mb': cache_file_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'stats': stats
        }
    
    def list_analyzed_images(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List analyzed images with optional limit.
        
        Args:
            limit: Maximum number of images to return
            
        Returns:
            List of analyzed image information
        """
        all_images = self.cache.get_analyzed_images()
        return all_images[:limit] if limit else all_images
    
    def find_duplicate_images(self) -> List[List[str]]:
        """
        Find duplicate images based on file content hash.
        
        Returns:
            List of groups of duplicate image paths
        """
        # Group images by their content hash
        content_groups = {}
        
        for entry in self.cache.get_analyzed_images():
            image_path = Path(entry['path'])
            if image_path.exists():
                try:
                    # Use file size and modification time as content hash
                    stat = image_path.stat()
                    content_hash = f"{stat.st_size}:{stat.st_mtime}"
                    
                    if content_hash not in content_groups:
                        content_groups[content_hash] = []
                    content_groups[content_hash].append(str(image_path))
                except Exception as e:
                    logger.warning(f"Error processing {image_path}: {e}")
        
        # Return groups with more than one image
        duplicates = [group for group in content_groups.values() if len(group) > 1]
        return duplicates
    
    def cleanup_invalid_entries(self) -> int:
        """
        Remove cache entries for images that no longer exist.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned_count = 0
        invalid_hashes = []
        
        for hash_key, entry in self.cache.cache_data.items():
            image_path = Path(entry['path'])
            if not image_path.exists():
                invalid_hashes.append(hash_key)
                cleaned_count += 1
        
        # Remove invalid entries
        for hash_key in invalid_hashes:
            del self.cache.cache_data[hash_key]
        
        if cleaned_count > 0:
            self.cache.save_cache()
            logger.info(f"Cleaned up {cleaned_count} invalid cache entries")
        
        return cleaned_count
    
    def export_cache_report(self, output_file: Path = None) -> Path:
        """
        Export detailed cache report to JSON file.
        
        Args:
            output_file: Output file path (default: cache_report.json)
            
        Returns:
            Path to exported report file
        """
        if output_file is None:
            output_file = self.cache_dir / f"cache_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'cache_info': self.get_cache_info(),
            'analyzed_images': self.list_analyzed_images(),
            'duplicates': self.find_duplicate_images(),
            'summary': {
                'total_analyzed': len(self.cache.cache_data),
                'duplicate_groups': len(self.find_duplicate_images()),
                'cache_hit_rate': self.cache.get_cache_stats().get('hit_rate', 0)
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Cache report exported to: {output_file}")
        return output_file
    
    def clear_old_entries(self, days_old: int = 30) -> int:
        """
        Clear cache entries older than specified days.
        
        Args:
            days_old: Remove entries older than this many days
            
        Returns:
            Number of entries removed
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_count = 0
        old_hashes = []
        
        for hash_key, entry in self.cache.cache_data.items():
            try:
                analyzed_at = datetime.fromisoformat(entry['analyzed_at'])
                if analyzed_at < cutoff_date:
                    old_hashes.append(hash_key)
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Error parsing date for {hash_key}: {e}")
                # Remove entries with invalid dates
                old_hashes.append(hash_key)
                removed_count += 1
        
        # Remove old entries
        for hash_key in old_hashes:
            del self.cache.cache_data[hash_key]
        
        if removed_count > 0:
            self.cache.save_cache()
            logger.info(f"Removed {removed_count} entries older than {days_old} days")
        
        return removed_count
    
    def optimize_cache(self) -> Dict[str, int]:
        """
        Perform comprehensive cache optimization.
        
        Returns:
            Dictionary with optimization results
        """
        results = {
            'invalid_entries_removed': 0,
            'old_entries_removed': 0,
            'duplicates_found': 0
        }
        
        # Remove invalid entries
        results['invalid_entries_removed'] = self.cleanup_invalid_entries()
        
        # Remove old entries (older than 30 days)
        results['old_entries_removed'] = self.clear_old_entries(30)
        
        # Count duplicates
        results['duplicates_found'] = len(self.find_duplicate_images())
        
        logger.info(f"Cache optimization completed: {results}")
        return results
    
    def backup_cache(self, backup_file: Path = None) -> Path:
        """
        Create backup of current cache.
        
        Args:
            backup_file: Backup file path (default: cache_backup.json)
            
        Returns:
            Path to backup file
        """
        if backup_file is None:
            backup_file = self.cache_dir / f"cache_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save current cache
        self.cache.save_cache()
        
        # Copy to backup location
        import shutil
        shutil.copy2(self.cache.cache_file, backup_file)
        
        logger.info(f"Cache backed up to: {backup_file}")
        return backup_file
    
    def restore_cache(self, backup_file: Path) -> bool:
        """
        Restore cache from backup file.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if restore was successful
        """
        try:
            import shutil
            shutil.copy2(backup_file, self.cache.cache_file)
            
            # Reload cache
            self.cache._load_cache()
            
            logger.info(f"Cache restored from: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore cache: {e}")
            return False


def main():
    """Command-line interface for cache management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage image analysis cache')
    parser.add_argument('--info', action='store_true', help='Show cache information')
    parser.add_argument('--cleanup', action='store_true', help='Clean up invalid entries')
    parser.add_argument('--optimize', action='store_true', help='Optimize cache')
    parser.add_argument('--export', type=str, help='Export cache report to file')
    parser.add_argument('--backup', type=str, help='Create cache backup')
    parser.add_argument('--restore', type=str, help='Restore cache from backup')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create cache manager
    manager = CacheManager()
    
    if args.info:
        info = manager.get_cache_info()
        print(f"Cache Information:")
        print(f"  File: {info['cache_file']}")
        print(f"  Size: {info['cache_file_size_mb']:.2f} MB")
        print(f"  Entries: {info['stats']['total_entries']}")
        print(f"  Hit Rate: {info['stats']['hit_rate']:.2%}")
    
    if args.cleanup:
        cleaned = manager.cleanup_invalid_entries()
        print(f"Cleaned up {cleaned} invalid entries")
    
    if args.optimize:
        results = manager.optimize_cache()
        print(f"Optimization results: {results}")
    
    if args.export:
        report_file = manager.export_cache_report(Path(args.export))
        print(f"Report exported to: {report_file}")
    
    if args.backup:
        backup_file = manager.backup_cache(Path(args.backup))
        print(f"Cache backed up to: {backup_file}")
    
    if args.restore:
        success = manager.restore_cache(Path(args.restore))
        print(f"Restore {'successful' if success else 'failed'}")


if __name__ == "__main__":
    main()
