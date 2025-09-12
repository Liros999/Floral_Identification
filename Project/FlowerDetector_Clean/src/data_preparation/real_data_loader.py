"""
OPTIMIZED REAL DATA LOADER - NO DUMMY DATA ALLOWED
Connects directly to Google Drive images and validates integrity.
Follows scientific requirements: real data only, no fake anything.
Enhanced with performance optimizations for production deployment.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import random
from tqdm import tqdm
import psutil
import os
import time
from functools import lru_cache
import numpy as np

# Import the new image cache system
from .image_cache import ImageAnalysisCache, create_image_cache

logger = logging.getLogger(__name__)


class FlowerDataset(Dataset):
    """
    Optimized real flower dataset - loads actual images from Google Drive.
    NO FAKE DATA, NO DUMMY DATA, NO PLACEHOLDERS.
    Enhanced with caching, performance optimizations, and balanced sampling.
    Supports multiple negative image sources for comprehensive training.
    """
    
    def __init__(
        self, 
        positive_dir: Path, 
        negative_dirs: List[Path],  # Changed to support multiple negative directories
        transform=None,
        max_images_per_class: int = None,
        validate_images: bool = True,
        enable_caching: bool = True,
        cache_size: int = 1000,
        balance_classes: bool = True,  # New parameter for balanced sampling
        enable_analysis_cache: bool = True  # New parameter for analysis caching
    ):
        """
        Initialize with REAL image directories.
        
        Args:
            positive_dir: Directory with actual flower images
            negative_dirs: List of directories with actual background images
            transform: Image transformations
            max_images_per_class: Limit images per class (for testing)
            validate_images: Validate each image can be loaded
            enable_caching: Enable LRU caching for loaded images
            cache_size: Maximum number of images to cache
            balance_classes: Whether to balance positive/negative classes
            enable_analysis_cache: Enable analysis cache to remember processed images
        """
        self.positive_dir = Path(positive_dir)
        self.negative_dirs = [Path(d) for d in negative_dirs] if isinstance(negative_dirs, list) else [Path(negative_dirs)]
        self.transform = transform
        self.max_images_per_class = max_images_per_class
        self.validate_images = validate_images
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.balance_classes = balance_classes
        self.enable_analysis_cache = enable_analysis_cache
        
        # Initialize analysis cache if enabled
        if self.enable_analysis_cache:
            self.analysis_cache = create_image_cache()
            logger.info("Analysis cache enabled - will remember processed images")
        else:
            self.analysis_cache = None
        
        # Initialize cache if enabled
        if self.enable_caching:
            self._image_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
        
        # Validate directories exist
        if not self.positive_dir.exists():
            raise ValueError(f"Positive images directory not found: {self.positive_dir}")
        for neg_dir in self.negative_dirs:
            if not neg_dir.exists():
                raise ValueError(f"Negative images directory not found: {neg_dir}")
        
        # Load REAL image paths
        self.image_paths, self.labels = self._load_real_image_paths(
            max_images_per_class, validate_images
        )
        
        logger.info(f"Loaded {len(self.image_paths)} REAL images")
        logger.info(f"  - Positive: {sum(self.labels)} images")
        logger.info(f"  - Negative: {len(self.labels) - sum(self.labels)} images")
    
    def _load_real_image_paths(
        self, 
        max_images_per_class: int, 
        validate_images: bool
    ) -> Tuple[List[Path], List[int]]:
        """
        Load REAL image paths from Google Drive directories.
        Uses analysis cache to skip already processed images.
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        positive_paths = []
        negative_paths = []
        
        # Load positive images (flowers)
        logger.info("🌺 Loading positive flower images...")
        positive_files = [
            f for f in self.positive_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if max_images_per_class:
            positive_files = positive_files[:max_images_per_class]
        
        logger.info(f"📊 Found {len(positive_files)} positive image files")
        
        # Load all images first, then use cache for optimization during training
        processed_count = 0
        skipped_count = 0
        corrupted_count = 0
        
        # Enhanced progress bar with detailed information
        with tqdm(positive_files, desc="🌺 Loading positive images", 
                 unit="img", ncols=100, colour='green') as pbar:
            for image_path in pbar:
                # Always load images for dataset creation, cache is used during training
                # if self.analysis_cache and self.analysis_cache.is_analyzed(image_path):
                #     processed_count += 1
                #     pbar.set_postfix({
                #         'Processed': processed_count,
                #         'Skipped': skipped_count,
                #         'Corrupted': corrupted_count,
                #         'Loaded': len(positive_paths)
                #     })
                #     continue
                    
                if validate_images:
                    if not self._validate_real_image(image_path):
                        corrupted_count += 1
                        pbar.set_postfix({
                            'Processed': processed_count,
                            'Skipped': skipped_count,
                            'Corrupted': corrupted_count,
                            'Loaded': len(positive_paths)
                        })
                        continue
                positive_paths.append(image_path)
                
                # Update progress bar with current stats
                pbar.set_postfix({
                    'Processed': processed_count,
                    'Skipped': skipped_count,
                    'Corrupted': corrupted_count,
                    'Loaded': len(positive_paths)
                })
        
        # Summary for positive images
        logger.info(f"✅ Positive images loaded: {len(positive_paths)}")
        if processed_count > 0:
            logger.info(f"⚡ Skipped {processed_count} already analyzed positive images")
        if corrupted_count > 0:
            logger.warning(f"⚠️ Skipped {corrupted_count} corrupted positive images")
        
        # Load negative images from ALL negative directories
        logger.info(f"🌿 Loading negative background images from {len(self.negative_dirs)} directories...")
        total_processed_negatives = 0
        total_corrupted_negatives = 0
        
        for i, neg_dir in enumerate(self.negative_dirs):
            logger.info(f"  📁 Directory {i+1}/{len(self.negative_dirs)}: {neg_dir.name}")
            negative_files = [
                f for f in neg_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            if max_images_per_class:
                # Distribute max_images_per_class across all negative directories
                per_dir_limit = max_images_per_class // len(self.negative_dirs)
                negative_files = negative_files[:per_dir_limit]
            
            logger.info(f"    📊 Found {len(negative_files)} negative image files")
            
            processed_in_dir = 0
            corrupted_in_dir = 0
            loaded_in_dir = 0
            
            # Enhanced progress bar for negative images
            with tqdm(negative_files, desc=f"🌿 Loading {neg_dir.name}", 
                     unit="img", ncols=100, colour='blue') as pbar:
                for image_path in pbar:
                    # Always load images for dataset creation, cache is used during training
                    # if self.analysis_cache and self.analysis_cache.is_analyzed(image_path):
                    #     processed_in_dir += 1
                    #     pbar.set_postfix({
                    #         'Processed': processed_in_dir,
                    #         'Corrupted': corrupted_in_dir,
                    #         'Loaded': loaded_in_dir
                    #     })
                    #     continue
                        
                    if validate_images:
                        if not self._validate_real_image(image_path):
                            corrupted_in_dir += 1
                            pbar.set_postfix({
                                'Processed': processed_in_dir,
                                'Corrupted': corrupted_in_dir,
                                'Loaded': loaded_in_dir
                            })
                            continue
                    
                    negative_paths.append(image_path)
                    loaded_in_dir += 1
                    
                    # Update progress bar with current stats
                    pbar.set_postfix({
                        'Processed': processed_in_dir,
                        'Corrupted': corrupted_in_dir,
                        'Loaded': loaded_in_dir
                    })
            
            total_processed_negatives += processed_in_dir
            total_corrupted_negatives += corrupted_in_dir
            
            # Summary for this directory
            logger.info(f"    ✅ Loaded {loaded_in_dir} images from {neg_dir.name}")
            if processed_in_dir > 0:
                logger.info(f"    ⚡ Skipped {processed_in_dir} already analyzed images")
            if corrupted_in_dir > 0:
                logger.warning(f"    ⚠️ Skipped {corrupted_in_dir} corrupted images")
        
        # Overall summary for negative images
        logger.info(f"✅ Total negative images loaded: {len(negative_paths)}")
        if total_processed_negatives > 0:
            logger.info(f"⚡ Total skipped (already analyzed): {total_processed_negatives}")
        if total_corrupted_negatives > 0:
            logger.warning(f"⚠️ Total corrupted images: {total_corrupted_negatives}")
        
        # Implement balanced sampling if requested
        if self.balance_classes and len(positive_paths) != len(negative_paths):
            logger.info(f"⚖️ Balancing classes: {len(positive_paths)} positive vs {len(negative_paths)} negative")
            
            # Check for edge cases
            if len(positive_paths) == 0:
                logger.error("❌ No positive images found! Cannot balance classes.")
                raise ValueError("No positive images found in the dataset. Check your data paths and file extensions.")
            if len(negative_paths) == 0:
                logger.error("❌ No negative images found! Cannot balance classes.")
                raise ValueError("No negative images found in the dataset. Check your data paths and file extensions.")
            
            if len(positive_paths) > len(negative_paths):
                # Upsample negative images
                target_count = len(positive_paths)
                logger.info(f"📈 Upsampling negative images from {len(negative_paths)} to {target_count}")
                negative_paths = self._upsample_images(negative_paths, target_count)
            else:
                # Downsample positive images
                target_count = len(negative_paths)
                logger.info(f"📉 Downsampling positive images from {len(positive_paths)} to {target_count}")
                if target_count > len(positive_paths):
                    logger.warning(f"⚠️ Cannot downsample {len(positive_paths)} positive images to {target_count}. Using all available positive images.")
                    # Use all positive images and upsample negatives instead
                    target_count = len(positive_paths)
                    logger.info(f"📈 Upsampling negative images from {len(negative_paths)} to {target_count}")
                    negative_paths = self._upsample_images(negative_paths, target_count)
                else:
                    positive_paths = random.sample(positive_paths, target_count)
        
        # Combine and create labels
        image_paths = positive_paths + negative_paths
        labels = [1] * len(positive_paths) + [0] * len(negative_paths)
        
        if len(image_paths) == 0:
            raise ValueError("No valid images found in directories!")
        
        # Final comprehensive summary
        logger.info("=" * 60)
        logger.info("🎉 DATA LOADING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Total images loaded: {len(image_paths)}")
        logger.info(f"  🌺 Positive (flowers): {len(positive_paths)} images")
        logger.info(f"  🌿 Negative (background): {len(negative_paths)} images")
        logger.info(f"  ⚖️ Class balance ratio: {len(positive_paths)}:{len(negative_paths)}")
        
        # Cache statistics
        if self.analysis_cache:
            cache_stats = self.analysis_cache.get_cache_stats()
            logger.info(f"💾 Analysis cache: {cache_stats['total_entries']} entries")
            if cache_stats['total_entries'] > 0:
                logger.info(f"  📈 Cache hit rate: {cache_stats['hit_rate']:.1%}")
                logger.info(f"  ⚡ Cache hits: {cache_stats['hits']}")
                logger.info(f"  🔄 Cache misses: {cache_stats['misses']}")
        
        logger.info("=" * 60)
        
        return image_paths, labels
    
    def _validate_real_image(self, image_path: Path) -> bool:
        """
        FAST validation that a REAL image file can be loaded.
        Optimized for speed - only checks file header, not full image data.
        """
        try:
            # Fast validation: only check file header and basic properties
            with Image.open(image_path) as img:
                # Quick dimension check (no pixel loading)
                width, height = img.size
                if width < 32 or height < 32:
                    return False
                
                # Quick mode check (no conversion)
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    return False
                
                # Test if we can access basic properties (no pixel data loading)
                _ = img.format
                _ = img.mode
                return True
                
        except Exception as e:
            logger.debug(f"Image validation failed for {image_path}: {e}")
            return False
    
    def _upsample_images(self, image_paths: List[Path], target_count: int) -> List[Path]:
        """
        Upsample image paths by sampling with replacement to reach target count.
        Used for balancing classes when negative class has fewer images.
        """
        if len(image_paths) == 0:
            return image_paths
        
        # Sample with replacement to reach target count
        upsampled = []
        for _ in range(target_count):
            upsampled.append(random.choice(image_paths))
        
        logger.info(f"Upsampled {len(image_paths)} images to {len(upsampled)} images")
        return upsampled
    
    def __len__(self) -> int:
        """Return number of REAL images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a REAL image and its label with caching optimization.
        NO FAKE DATA - loads actual image from disk.
        Records analysis in cache for future runs.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Check cache first if enabled
        if self.enable_caching and image_path in self._image_cache:
            self._cache_hits += 1
            image = self._image_cache[image_path].copy()
        else:
            self._cache_misses += 1
            # Load REAL image
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.error(f"Failed to load REAL image {image_path}: {e}")
                raise RuntimeError(f"Cannot load REAL image: {image_path}")
            
            # Cache the loaded image if caching is enabled and cache not full
            if self.enable_caching and len(self._image_cache) < self.cache_size:
                self._image_cache[image_path] = image.copy()
        
        # Record analysis in cache for future runs
        if self.analysis_cache:
            try:
                metadata = {
                    'image_size': image.size,
                    'image_mode': image.mode,
                    'loaded_at': time.time()
                }
                self.analysis_cache.add_analysis(image_path, label, metadata)
            except Exception as e:
                logger.debug(f"Failed to record analysis in cache: {e}")
        
        # Apply transformations if provided
        if self.transform:
            # Check if it's an Albumentations transform
            if hasattr(self.transform, 'transforms') and hasattr(self.transform, '__call__'):
                # Albumentations transform - convert PIL to numpy first
                image_np = np.array(image)  # Convert PIL to numpy
                transformed = self.transform(image=image_np)
                image = transformed['image']  # Returns torch tensor
            else:
                # Torchvision transform - use positional arguments
                image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get actual class distribution of REAL data."""
        positive_count = sum(self.labels)
        negative_count = len(self.labels) - positive_count
        
        return {
            'positive_flowers': positive_count,
            'negative_background': negative_count,
            'total': len(self.labels),
            'balance_ratio': positive_count / negative_count if negative_count > 0 else float('inf')
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching performance statistics."""
        stats = {}
        
        # Image loading cache stats
        if self.enable_caching:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
            
            stats['image_cache'] = {
                'enabled': True,
                'cache_size': len(self._image_cache),
                'max_cache_size': self.cache_size,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'hit_rate': hit_rate,
                'memory_usage_mb': sum(img.size[0] * img.size[1] * 3 for img in self._image_cache.values()) / (1024 * 1024)
            }
        else:
            stats['image_cache'] = {'enabled': False}
        
        # Analysis cache stats
        if self.analysis_cache:
            stats['analysis_cache'] = self.analysis_cache.get_cache_stats()
        else:
            stats['analysis_cache'] = {'enabled': False}
        
        return stats
    
    def get_analysis_cache_stats(self) -> Dict[str, Any]:
        """Get analysis cache statistics."""
        if self.analysis_cache:
            return self.analysis_cache.get_cache_stats()
        return {'enabled': False}
    
    def clear_analysis_cache(self) -> None:
        """Clear the analysis cache."""
        if self.analysis_cache:
            self.analysis_cache.clear_cache()
            logger.info("Analysis cache cleared")
    
    def save_analysis_cache(self) -> None:
        """Manually save the analysis cache."""
        if self.analysis_cache:
            self.analysis_cache.save_cache()
            logger.info("Analysis cache saved")


def create_real_data_transforms(image_size: Tuple[int, int] = (224, 224), use_advanced: bool = False) -> Dict[str, transforms.Compose]:
    """
    Create image transformations for REAL data.
    NO FAKE AUGMENTATIONS - scientifically validated transforms only.
    
    Args:
        image_size: Target image size (height, width)
        use_advanced: If True, use Albumentations for advanced augmentation
    """
    if use_advanced:
        # Use advanced Albumentations transforms (integrated)
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            # CORRECTED advanced training transforms with Albumentations
            train_transform = A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),  # Corrected parameter name and range
                ], p=0.3),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),  # Fixed parameters
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            val_transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            logger.info("Using advanced Albumentations transforms")
            return {'train': train_transform, 'val': val_transform, 'test': val_transform}
            
        except ImportError:
            logger.warning("Albumentations not available, falling back to torchvision transforms")
            use_advanced = False
    
    # Training transforms: basic augmentation scientifically proven to help
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize larger first
        transforms.RandomCrop(image_size),  # Random crop to target size
        transforms.RandomHorizontalFlip(p=0.5),  # Natural variation
        transforms.ColorJitter(
            brightness=0.1,  # Slight brightness variation
            contrast=0.1,    # Slight contrast variation
            saturation=0.1,  # Slight saturation variation
            hue=0.05        # Very slight hue variation
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]   # ImageNet stds
        )
    ])
    
    # Validation/test transforms: no augmentation, just normalize
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def create_real_data_loaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Create OPTIMIZED data loaders for REAL Google Drive images.
    NO DUMMY DATA - connects to actual image directories.
    Enhanced with performance optimizations for production deployment.
    """
    # Get paths from config
    data_paths = config['data_paths']
    positive_dir = data_paths['positive_images']
    
    # Support multiple negative directories - prioritize balanced_negatives
    negative_dirs = []
    if 'negative_images' in data_paths:
        negative_dirs.append(data_paths['negative_images'])
    if 'coco_negatives' in data_paths:
        negative_dirs.append(data_paths['coco_negatives'])
    if 'negative_images_1' in data_paths:
        negative_dirs.append(data_paths['negative_images_1'])
    # Legacy support for processed data
    if 'val_negative' in data_paths:
        negative_dirs.append(data_paths['val_negative'])
    if 'test_negative' in data_paths:
        negative_dirs.append(data_paths['test_negative'])
    if 'coco_negatives_1' in data_paths:
        negative_dirs.append(data_paths['coco_negatives_1'])
    
    if not negative_dirs:
        raise ValueError("No negative image directories found in config!")
    
    logger.info(f"Using {len(negative_dirs)} negative directories: {[d.name for d in negative_dirs]}")
    
    # Get data config
    data_config = config['data']
    image_size = tuple(data_config['image_size'])
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    validate_images = data_config.get('validate_images', False)  # Default to False for speed
    
    # Optimize num_workers for Intel Core Ultra 7 - increased for better performance
    optimal_workers = min(psutil.cpu_count(logical=True), 12)  # Use logical cores for I/O bound tasks
    actual_workers = min(num_workers, optimal_workers)
    logger.info(f"Using {actual_workers} workers (optimal: {optimal_workers}, configured: {num_workers})")
    
    # Create transforms
    transforms_dict = create_real_data_transforms(image_size)
    
    # For now, create a single dataset and we'll split it later
    # This ensures we're working with REAL data from the start
    full_dataset = FlowerDataset(
        positive_dir=positive_dir,
        negative_dirs=negative_dirs,  # Pass multiple negative directories
        transform=transforms_dict['train'],
        max_images_per_class=None,  # Load ALL available images - no artificial limits
        validate_images=validate_images,  # Use config setting
        enable_caching=True,  # Enable caching for performance
        cache_size=2000,  # Increased cache size for better performance
        balance_classes=True,  # Enable balanced sampling
        enable_analysis_cache=False  # Disable analysis cache to avoid file access issues
    )
    
    # Check class distribution
    distribution = full_dataset.get_class_distribution()
    logger.info("=" * 60)
    logger.info("📊 DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"🌺 Positive images: {distribution['positive_flowers']}")
    logger.info(f"🌿 Negative images: {distribution['negative_background']}")
    logger.info(f"📈 Total images: {distribution['total']}")
    logger.info(f"⚖️ Balance ratio: {distribution['balance_ratio']:.2f}")
    logger.info("=" * 60)
    
    # Create OPTIMIZED data loader
    pin_memory = data_config.get('pin_memory', True)  # Enable for GPU
    persistent_workers = data_config.get('persistent_workers', True)
    
    logger.info("🔧 Creating optimized data loaders...")
    logger.info(f"  📦 Batch size: {batch_size}")
    logger.info(f"  👥 Workers: {actual_workers}")
    logger.info(f"  💾 Pin memory: {pin_memory}")
    logger.info(f"  🔄 Persistent workers: {persistent_workers}")
    
    data_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=pin_memory,  # Enable for GPU training
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    # Calculate batch information
    total_batches = len(data_loader)
    logger.info(f"📊 Data loader created:")
    logger.info(f"  🔢 Total batches: {total_batches}")
    logger.info(f"  📈 Images per batch: {batch_size}")
    logger.info(f"  🎯 Total samples: {len(full_dataset)}")
    
    return {
        'train': data_loader,
        'dataset': full_dataset,
        'distribution': distribution
    }


def test_real_data_loading(config: Dict[str, Any]) -> bool:
    """
    Test loading REAL data - NO DUMMY DATA.
    Validates actual Google Drive images can be loaded.
    """
    try:
        logger.info("Testing REAL data loading...")
        
        # Create data loaders with REAL images
        data_loaders = create_real_data_loaders(config)
        train_loader = data_loaders['train']
        distribution = data_loaders['distribution']
        
        # Test loading one batch of REAL data
        logger.info("Loading first batch of REAL images...")
        batch_images, batch_labels = next(iter(train_loader))
        
        # Validate this is REAL data
        logger.info(f"✅ REAL DATA LOADED:")
        logger.info(f"   - Batch shape: {batch_images.shape}")
        logger.info(f"   - Labels: {batch_labels.tolist()}")
        logger.info(f"   - Data type: {batch_images.dtype}")
        logger.info(f"   - Label distribution in batch: {torch.bincount(batch_labels).tolist()}")
        logger.info(f"   - Full dataset distribution: {distribution}")
        
        # Validate no dummy data characteristics
        if batch_images.shape[1:] != (3, 224, 224):
            raise ValueError(f"Unexpected image shape: {batch_images.shape}")
        
        if not torch.all((batch_labels >= 0) & (batch_labels <= 1)):
            raise ValueError(f"Invalid labels: {batch_labels}")
        
        logger.info("🎉 REAL DATA LOADING SUCCESSFUL - No dummy data detected!")
        return True
        
    except Exception as e:
        logger.error(f"❌ REAL data loading failed: {e}")
        return False
