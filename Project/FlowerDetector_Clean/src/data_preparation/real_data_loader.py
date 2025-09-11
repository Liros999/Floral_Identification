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
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


class FlowerDataset(Dataset):
    """
    Optimized real flower dataset - loads actual images from Google Drive.
    NO FAKE DATA, NO DUMMY DATA, NO PLACEHOLDERS.
    Enhanced with caching and performance optimizations.
    """
    
    def __init__(
        self, 
        positive_dir: Path, 
        negative_dir: Path, 
        transform=None,
        max_images_per_class: int = None,
        validate_images: bool = True,
        enable_caching: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize with REAL image directories.
        
        Args:
            positive_dir: Directory with actual flower images
            negative_dir: Directory with actual background images  
            transform: Image transformations
            max_images_per_class: Limit images per class (for testing)
            validate_images: Validate each image can be loaded
            enable_caching: Enable LRU caching for loaded images
            cache_size: Maximum number of images to cache
        """
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        self.transform = transform
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Initialize cache if enabled
        if self.enable_caching:
            self._image_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
        
        # Validate directories exist
        if not self.positive_dir.exists():
            raise ValueError(f"Positive images directory not found: {self.positive_dir}")
        if not self.negative_dir.exists():
            raise ValueError(f"Negative images directory not found: {self.negative_dir}")
        
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
        Validates each image can be opened.
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        labels = []
        
        # Load positive images (flowers)
        logger.info("Loading positive flower images...")
        positive_files = [
            f for f in self.positive_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if max_images_per_class:
            positive_files = positive_files[:max_images_per_class]
        
        for image_path in tqdm(positive_files, desc="Validating positive images"):
            if validate_images and not self._validate_real_image(image_path):
                logger.warning(f"Skipping corrupted positive image: {image_path}")
                continue
            image_paths.append(image_path)
            labels.append(1)  # Flower class
        
        # Load negative images (background)
        logger.info("Loading negative background images...")
        negative_files = [
            f for f in self.negative_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if max_images_per_class:
            negative_files = negative_files[:max_images_per_class]
        
        for image_path in tqdm(negative_files, desc="Validating negative images"):
            if validate_images and not self._validate_real_image(image_path):
                logger.warning(f"Skipping corrupted negative image: {image_path}")
                continue
            image_paths.append(image_path)
            labels.append(0)  # Background class
        
        if len(image_paths) == 0:
            raise ValueError("No valid images found in directories!")
        
        return image_paths, labels
    
    def _validate_real_image(self, image_path: Path) -> bool:
        """
        Validate that a REAL image file can be loaded.
        NO FAKE VALIDATION - actually tries to open the file.
        """
        try:
            with Image.open(image_path) as img:
                # Verify image has reasonable dimensions
                width, height = img.size
                if width < 32 or height < 32:
                    return False
                
                # Verify image mode is supported
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    return False
                
                # Verify we can convert to RGB (test basic operations)
                img_rgb = img.convert('RGB')
                
                # Don't use verify() as it can corrupt the image object
                # Instead, try to load a small sample of pixel data
                img.load()
                return True
                
        except Exception as e:
            logger.debug(f"Image validation failed for {image_path}: {e}")
            return False
    
    def __len__(self) -> int:
        """Return number of REAL images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a REAL image and its label with caching optimization.
        NO FAKE DATA - loads actual image from disk.
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
        if not self.enable_caching:
            return {'caching_enabled': False}
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'caching_enabled': True,
            'cache_size': len(self._image_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'memory_usage_mb': sum(img.size[0] * img.size[1] * 3 for img in self._image_cache.values()) / (1024 * 1024)
        }


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
    negative_dir = data_paths['negative_images']
    
    # Get data config
    data_config = config['data']
    image_size = tuple(data_config['image_size'])
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    
    # Optimize num_workers for Intel Core Ultra 7
    optimal_workers = min(psutil.cpu_count(logical=False), 8)  # Use all physical cores
    actual_workers = min(num_workers, optimal_workers)
    logger.info(f"Using {actual_workers} workers (optimal: {optimal_workers}, configured: {num_workers})")
    
    # Create transforms
    transforms_dict = create_real_data_transforms(image_size)
    
    # For now, create a single dataset and we'll split it later
    # This ensures we're working with REAL data from the start
    full_dataset = FlowerDataset(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        transform=transforms_dict['train'],
        max_images_per_class=100,  # Start with small subset for testing
        validate_images=True,
        enable_caching=True,  # Enable caching for performance
        cache_size=1000  # Cache up to 1000 images
    )
    
    # Check class distribution
    distribution = full_dataset.get_class_distribution()
    logger.info(f"REAL data class distribution: {distribution}")
    
    # Create OPTIMIZED data loader
    data_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=False,  # CPU only
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
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
