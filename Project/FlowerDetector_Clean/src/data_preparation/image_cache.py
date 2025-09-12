"""
INTELLIGENT IMAGE ANALYSIS CACHE SYSTEM
Remembers which pictures have already been analyzed to prevent reprocessing.
Supports cache validation, cleanup, and performance optimization.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ImageAnalysisCache:
    """
    Intelligent cache system for image analysis results.
    Prevents reprocessing of already analyzed images.
    """
    
    def __init__(self, cache_file: Path, max_cache_size: int = 50000):
        """
        Initialize image analysis cache.
        
        Args:
            cache_file: Path to JSON cache file
            max_cache_size: Maximum number of entries to keep in cache
        """
        self.cache_file = Path(cache_file)
        self.max_cache_size = max_cache_size
        self.cache_data = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'validations': 0,
            'cleanups': 0,
            'last_updated': None
        }
        
        # Create cache directory if it doesn't exist
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"Image analysis cache initialized: {self.cache_file}")
        logger.info(f"Cache entries: {len(self.cache_data)}")
    
    def _load_cache(self) -> None:
        """Load cache from JSON file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache_data = data.get('images', {})
                    self.cache_stats = data.get('stats', self.cache_stats)
                logger.info(f"Loaded cache with {len(self.cache_data)} entries")
            else:
                logger.info("No existing cache found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, starting fresh")
            self.cache_data = {}
    
    def _save_cache(self) -> None:
        """Save cache to JSON file."""
        try:
            # Update cache statistics
            self.cache_stats['last_updated'] = datetime.now().isoformat()
            
            cache_data = {
                'images': self.cache_data,
                'stats': self.cache_stats,
                'metadata': {
                    'version': '1.0',
                    'created': datetime.now().isoformat(),
                    'max_size': self.max_cache_size
                }
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.cache_file)
            logger.debug(f"Cache saved: {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_image_hash(self, image_path: Path) -> str:
        """
        Generate unique hash for image based on path and modification time.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Unique hash string for the image
        """
        try:
            # Get file stats for hash
            stat = image_path.stat()
            # Combine path and modification time for uniqueness
            hash_input = f"{image_path.resolve()}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate hash for {image_path}: {e}")
            # Fallback to just path
            return hashlib.md5(str(image_path.resolve()).encode()).hexdigest()
    
    def is_analyzed(self, image_path: Path) -> bool:
        """
        Check if image has already been analyzed.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image has been analyzed and is still valid
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return False
            
            image_hash = self._get_image_hash(image_path)
            
            if image_hash in self.cache_data:
                # Validate that the cached entry is still valid
                if self._validate_cache_entry(image_path, image_hash):
                    self.cache_stats['hits'] += 1
                    return True
                else:
                    # Remove invalid entry
                    del self.cache_data[image_hash]
                    self.cache_stats['cleanups'] += 1
            
            self.cache_stats['misses'] += 1
            return False
            
        except Exception as e:
            logger.warning(f"Error checking if image is analyzed: {e}")
            return False
    
    def _validate_cache_entry(self, image_path: Path, image_hash: str) -> bool:
        """
        Validate that a cached entry is still valid.
        
        Args:
            image_path: Path to image file
            image_hash: Hash of the image
            
        Returns:
            True if cache entry is still valid
        """
        try:
            self.cache_stats['validations'] += 1
            
            # Check if file still exists
            if not image_path.exists():
                return False
            
            # Check if file has been modified
            current_hash = self._get_image_hash(image_path)
            if current_hash != image_hash:
                return False
            
            # Check if cached data is complete
            cached_data = self.cache_data[image_hash]
            required_fields = ['path', 'label', 'analyzed_at', 'file_size']
            if not all(field in cached_data for field in required_fields):
                return False
            
            # Check if file size matches
            if cached_data['file_size'] != image_path.stat().st_size:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating cache entry: {e}")
            return False
    
    def add_analysis(self, image_path: Path, label: int, metadata: Dict[str, Any] = None) -> None:
        """
        Add image analysis result to cache.
        
        Args:
            image_path: Path to analyzed image
            label: Image label (0 for negative, 1 for positive)
            metadata: Additional metadata about the analysis
        """
        try:
            image_path = Path(image_path)
            image_hash = self._get_image_hash(image_path)
            
            # Prepare analysis data
            analysis_data = {
                'path': str(image_path.resolve()),
                'label': label,
                'analyzed_at': datetime.now().isoformat(),
                'file_size': image_path.stat().st_size,
                'file_mtime': image_path.stat().st_mtime,
                'metadata': metadata or {}
            }
            
            # Add to cache
            self.cache_data[image_hash] = analysis_data
            
            # Cleanup if cache is too large
            if len(self.cache_data) > self.max_cache_size:
                self._cleanup_cache()
            
            # Save cache periodically (every 100 additions)
            if len(self.cache_data) % 100 == 0:
                self._save_cache()
            
            logger.debug(f"Added analysis to cache: {image_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to add analysis to cache: {e}")
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries to maintain size limit."""
        try:
            if len(self.cache_data) <= self.max_cache_size:
                return
            
            # Sort by analysis time (oldest first)
            sorted_entries = sorted(
                self.cache_data.items(),
                key=lambda x: x[1].get('analyzed_at', ''),
                reverse=False
            )
            
            # Remove oldest entries
            entries_to_remove = len(self.cache_data) - self.max_cache_size
            for i in range(entries_to_remove):
                hash_key, _ = sorted_entries[i]
                del self.cache_data[hash_key]
            
            self.cache_stats['cleanups'] += 1
            logger.info(f"Cleaned up {entries_to_remove} old cache entries")
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
    
    def get_analysis(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result for image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Cached analysis data or None if not found
        """
        try:
            image_path = Path(image_path)
            image_hash = self._get_image_hash(image_path)
            
            if image_hash in self.cache_data:
                if self._validate_cache_entry(image_path, image_hash):
                    return self.cache_data[image_hash]
                else:
                    del self.cache_data[image_hash]
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting analysis from cache: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache_data),
            'max_size': self.max_cache_size,
            'hit_rate': hit_rate,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'validations': self.cache_stats['validations'],
            'cleanups': self.cache_stats['cleanups'],
            'last_updated': self.cache_stats['last_updated']
        }
    
    def clear_cache(self) -> None:
        """Clear all cache data."""
        self.cache_data = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'validations': 0,
            'cleanups': 0,
            'last_updated': None
        }
        self._save_cache()
        logger.info("Cache cleared")
    
    def save_cache(self) -> None:
        """Manually save cache to disk."""
        self._save_cache()
        logger.info("Cache saved manually")
    
    def get_analyzed_images(self) -> List[Dict[str, Any]]:
        """Get list of all analyzed images."""
        return list(self.cache_data.values())
    
    def get_analyzed_paths(self) -> List[Path]:
        """Get list of all analyzed image paths."""
        return [Path(entry['path']) for entry in self.cache_data.values()]
    
    def is_path_analyzed(self, image_path: Path) -> bool:
        """Check if specific path has been analyzed (by path, not hash)."""
        try:
            image_path = Path(image_path).resolve()
            for entry in self.cache_data.values():
                if Path(entry['path']).resolve() == image_path:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error checking if path is analyzed: {e}")
            return False


def create_image_cache(cache_dir: Path = None) -> ImageAnalysisCache:
    """
    Factory function to create image analysis cache.
    
    Args:
        cache_dir: Directory for cache file (default: project cache directory)
        
    Returns:
        Configured ImageAnalysisCache instance
    """
    if cache_dir is None:
        # Default to project cache directory
        project_root = Path(__file__).resolve().parents[2]
        cache_dir = project_root / 'cache'
    
    cache_file = cache_dir / 'image_analysis_cache.json'
    return ImageAnalysisCache(cache_file)


# Example usage and testing
if __name__ == "__main__":
    # Test the cache system
    cache = create_image_cache()
    
    # Test with a sample image path
    test_path = Path("test_image.jpg")
    
    # Check if analyzed
    print(f"Is analyzed: {cache.is_analyzed(test_path)}")
    
    # Add analysis
    cache.add_analysis(test_path, label=1, metadata={'confidence': 0.95})
    
    # Check again
    print(f"Is analyzed: {cache.is_analyzed(test_path)}")
    
    # Get stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
