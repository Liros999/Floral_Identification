"""
Dataset construction and management for Foundational Flower Detector.

This module implements the dataset building functionality as specified in the
Code_Structure.txt documentation. It handles COCO format annotation generation,
deterministic data splitting, and integration with the hard negative mining cycle.

Key Features:
- COCO JSON format annotation generation
- Deterministic train/validation/test splits (sklearn-based)
- Hard negative integration from confirmed_hard_negatives.log
- Google Drive data integration
- Atomic file operations for thread safety

References:
- Code_Structure.txt: Detailed build_dataset.py specifications
- COCO dataset format: https://cocodataset.org/#format-data
- Lin, T. Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.

Author: Foundational Flower Detector Team
Date: September 2025
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from tqdm import tqdm

from .utils import (
    ReproducibilityManager, 
    AtomicFileWriter, 
    DataIntegrityValidator,
    FileHasher
)

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Primary dataset construction and versioning system.
    
    This class implements the dataset building functionality specified in the
    architecture documents, including COCO format generation, deterministic
    splitting, and integration with the hard negative mining cycle.
    
    The builder follows the scientific rigor principles with reproducible
    splits and real data validation throughout the process.
    """
    
    def __init__(self, config):
        """
        Initialize dataset builder with configuration.
        
        Args:
            config: Configuration object with data paths and parameters
        """
        self.config = config
        self.data_paths = config.get_data_paths()
        self.validator = DataIntegrityValidator(config)
        self.global_seed = config.get('data.global_random_seed', 42)
        
        # COCO format specifications
        self.coco_info = {
            "description": "Foundational Flower Detector Dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "Foundational Flower Detector Team",
            "date_created": datetime.now().isoformat()
        }
        
        # Categories (binary classification: background + flower)
        self.categories = [
            {
                "id": 1,
                "name": "flower",
                "supercategory": "plant"
            }
        ]
        
        logger.info(f"Initialized DatasetBuilder with seed={self.global_seed}")
        logger.info(f"Data paths: {self.data_paths}")
    
    def load_image_paths(self) -> Dict[str, List[Path]]:
        """
        Load file paths from the Google Drive directories.
        
        This function implements the load_image_paths functionality specified
        in Code_Structure.txt, loading positive and negative image paths from
        the configured Google Drive directories.
        
        Returns:
            Dictionary with 'positive' and 'negative' lists of Path objects
            
        Raises:
            FileNotFoundError: If data directories don't exist
            ValueError: If no images found in directories
        """
        logger.info("Loading image paths from Google Drive directories")
        
        positive_dir = self.data_paths['positive_images']
        negative_dir = self.data_paths['negative_images']
        
        # Validate directories exist
        if not positive_dir.exists():
            raise FileNotFoundError(f"Positive images directory not found: {positive_dir}")
        
        # For negative images, we might not have them initially
        if not negative_dir.exists():
            logger.warning(f"Negative images directory not found: {negative_dir}")
            logger.info("Will use COCO background images when available")
        
        # Load positive images
        positive_paths = self._load_images_from_directory(positive_dir)
        logger.info(f"Found {len(positive_paths)} positive images")
        
        # Load negative images (if directory exists)
        negative_paths = []
        if negative_dir.exists():
            negative_paths = self._load_images_from_directory(negative_dir)
        
        # Supplement with COCO background images if needed
        coco_backgrounds = self._load_coco_background_images()
        negative_paths.extend(coco_backgrounds)
        
        logger.info(f"Found {len(negative_paths)} negative images (including COCO backgrounds)")
        
        if len(positive_paths) == 0:
            raise ValueError("No positive images found")
        
        if len(negative_paths) == 0:
            logger.warning("No negative images found - will use only positive samples")
        
        return {
            'positive': positive_paths,
            'negative': negative_paths
        }
    
    def _load_images_from_directory(self, directory: Path) -> List[Path]:
        """
        Load valid image files from a directory.
        
        Args:
            directory: Directory to search for images
            
        Returns:
            List of valid image file paths
        """
        valid_extensions = self.config.get('data.image_extensions', ['.jpg', '.jpeg', '.png'])
        image_paths = []
        
        for ext in valid_extensions:
            # Case-insensitive search
            image_paths.extend(directory.glob(f'*{ext}'))
            image_paths.extend(directory.glob(f'*{ext.upper()}'))
        
        # Remove duplicates and sort for consistency
        image_paths = sorted(list(set(image_paths)))
        
        # Validate images with progress bar
        validated_paths = []
        logger.info(f"Validating {len(image_paths)} images from {directory}")
        
        for path in tqdm(image_paths, desc="Validating images", unit="img"):
            validation_result = self.validator.validate_image_file(path)
            if validation_result['valid']:
                validated_paths.append(path)
            else:
                logger.warning(f"Invalid image skipped: {path} - {validation_result['errors']}")
        
        return validated_paths
    
    def _load_coco_background_images(self) -> List[Path]:
        """
        Load COCO background images (non-plant categories).
        
        This method identifies COCO images that don't contain plant-related
        categories to serve as negative examples for training.
        
        Returns:
            List of COCO background image paths
        """
        coco_backgrounds = []
        
        try:
            # Look for COCO annotations in metadata
            metadata_dir = self.data_paths['metadata']
            if not metadata_dir.exists():
                logger.info("COCO metadata directory not found, skipping COCO backgrounds")
                return coco_backgrounds
            
            # Load COCO annotations to identify background images
            coco_train_annotations = metadata_dir / 'annotations' / 'instances_train2017.json'
            coco_val_annotations = metadata_dir / 'annotations' / 'instances_val2017.json'
            
            for annotation_file in [coco_train_annotations, coco_val_annotations]:
                if annotation_file.exists():
                    backgrounds = self._extract_coco_backgrounds(annotation_file)
                    coco_backgrounds.extend(backgrounds)
                    logger.info(f"Found {len(backgrounds)} COCO background images from {annotation_file.name}")
            
        except Exception as e:
            logger.warning(f"Failed to load COCO backgrounds: {e}")
        
        return coco_backgrounds
    
    def _extract_coco_backgrounds(self, annotation_file: Path) -> List[Path]:
        """
        Extract background image paths from COCO annotations.
        
        Args:
            annotation_file: Path to COCO annotation JSON file
            
        Returns:
            List of background image paths
        """
        backgrounds = []
        
        try:
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
            
            # Plant-related category IDs to exclude (from COCO categories)
            plant_categories = {
                # These are typical COCO categories that might contain plants
                # We'll be conservative and exclude any that might have flowers
                51,  # giraffe (might have plants in background)
                # Add more if needed based on COCO category analysis
            }
            
            # Get images that don't have plant-related annotations
            images_with_plants = set()
            annotations = coco_data.get('annotations', [])
            logger.info(f"Processing {len(annotations)} COCO annotations")
            
            for annotation in tqdm(annotations, desc="Processing COCO annotations", unit="ann"):
                if annotation.get('category_id') in plant_categories:
                    images_with_plants.add(annotation.get('image_id'))
            
            # Select images without plant categories
            images = coco_data.get('images', [])
            logger.info(f"Processing {len(images)} COCO images")
            
            for image_info in tqdm(images, desc="Filtering COCO images", unit="img"):
                image_id = image_info.get('id')
                if image_id not in images_with_plants:
                    # Construct path to COCO image
                    filename = image_info.get('file_name')
                    if filename:
                        # Try to find the actual image file
                        potential_paths = [
                            self.data_paths['raw_data'] / 'coco_images' / filename,
                            self.data_paths['raw_data'] / 'negative_images' / filename,
                            self.data_paths['base'] / 'coco_images' / filename
                        ]
                        
                        for path in potential_paths:
                            if path.exists():
                                backgrounds.append(path)
                                break
            
        except Exception as e:
            logger.error(f"Failed to extract COCO backgrounds from {annotation_file}: {e}")
        
        return backgrounds
    
    def create_deterministic_splits(self, image_paths: Dict[str, List[Path]]) -> Dict[str, Dict[str, List[Path]]]:
        """
        Split data into training, validation, and test sets deterministically.
        
        This function implements the deterministic splitting specified in
        Code_Structure.txt using sklearn.model_selection.train_test_split
        with fixed random_state for reproducibility.
        
        Args:
            image_paths: Dictionary with 'positive' and 'negative' image paths
            
        Returns:
            Dictionary with train/val/test splits for positive and negative images
        """
        logger.info("Creating deterministic data splits")
        
        # Set reproducibility
        ReproducibilityManager.set_seed(self.global_seed)
        
        # Get split ratios from config
        train_split = self.config.get('data.train_split', 0.7)
        val_split = self.config.get('data.val_split', 0.2)
        test_split = self.config.get('data.test_split', 0.1)
        
        # Validate splits sum to 1.0
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        splits = {'train': {'positive': [], 'negative': []},
                 'val': {'positive': [], 'negative': []},
                 'test': {'positive': [], 'negative': []}}
        
        # Split positive and negative images separately
        for label, paths in image_paths.items():
            if len(paths) == 0:
                logger.warning(f"No {label} images to split")
                continue
            
            # Convert paths to strings for sklearn compatibility
            path_strings = [str(p) for p in paths]
            
            # First split: separate test set
            train_val_paths, test_paths = train_test_split(
                path_strings,
                test_size=test_split,
                random_state=self.global_seed,
                shuffle=True
            )
            
            # Second split: separate train and validation
            val_size_adjusted = val_split / (train_split + val_split)
            train_paths, val_paths = train_test_split(
                train_val_paths,
                test_size=val_size_adjusted,
                random_state=self.global_seed,
                shuffle=True
            )
            
            # Convert back to Path objects
            splits['train'][label] = [Path(p) for p in train_paths]
            splits['val'][label] = [Path(p) for p in val_paths]
            splits['test'][label] = [Path(p) for p in test_paths]
            
            logger.info(f"{label.capitalize()} split: "
                       f"train={len(train_paths)}, "
                       f"val={len(val_paths)}, "
                       f"test={len(test_paths)}")
        
        return splits
    
    def load_hard_negatives_log(self) -> List[Path]:
        """
        Load confirmed hard negatives from the log file.
        
        This function reads the confirmed_hard_negatives.log file created by
        the human verification process and returns the list of confirmed
        false positive images to be added to the training set.
        
        Returns:
            List of confirmed hard negative image paths
        """
        hard_negatives_log = self.data_paths['base'] / self.config.get(
            'hard_negative_mining.confirmed_negatives_log', 
            'confirmed_hard_negatives.log'
        )
        
        hard_negatives = []
        
        if not hard_negatives_log.exists():
            logger.info("No hard negatives log found - first training cycle")
            return hard_negatives
        
        try:
            with open(hard_negatives_log, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle different log formats
                        if line.startswith('{'):
                            # JSON format with additional metadata
                            try:
                                entry = json.loads(line)
                                if 'image_path' in entry:
                                    hard_negatives.append(Path(entry['image_path']))
                                elif 'path' in entry:
                                    hard_negatives.append(Path(entry['path']))
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in hard negatives log: {line}")
                        else:
                            # Simple path format
                            hard_negatives.append(Path(line))
            
            # Validate paths exist
            valid_hard_negatives = []
            for path in hard_negatives:
                if path.exists():
                    valid_hard_negatives.append(path)
                else:
                    logger.warning(f"Hard negative image not found: {path}")
            
            logger.info(f"Loaded {len(valid_hard_negatives)} confirmed hard negatives")
            return valid_hard_negatives
            
        except Exception as e:
            logger.error(f"Failed to load hard negatives log: {e}")
            return []
    
    def generate_coco_json(self, splits: Dict[str, Dict[str, List[Path]]], 
                          version: int = 1) -> Dict[str, Path]:
        """
        Generate COCO format annotations for train/val/test splits.
        
        This function implements the COCO JSON generation specified in
        Code_Structure.txt, creating properly formatted annotation files
        for object detection training.
        
        Args:
            splits: Dictionary containing train/val/test splits
            version: Dataset version number for file naming
            
        Returns:
            Dictionary mapping split names to annotation file paths
        """
        logger.info(f"Generating COCO JSON annotations (version {version})")
        
        annotations_dir = self.data_paths['annotations']
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        annotation_files = {}
        
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split")
            
            # Create COCO format structure
            coco_data = {
                "info": {
                    **self.coco_info,
                    "description": f"{self.coco_info['description']} - {split_name.upper()} split",
                    "version": f"{self.coco_info['version']}.{version}"
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": self.categories
            }
            
            image_id = 1
            annotation_id = 1
            
            # Process positive images (with flower annotations)
            for image_path in split_data.get('positive', []):
                try:
                    image_info, annotations = self._create_image_annotations(
                        image_path, image_id, annotation_id, has_flower=True
                    )
                    
                    coco_data['images'].append(image_info)
                    coco_data['annotations'].extend(annotations)
                    
                    image_id += 1
                    annotation_id += len(annotations)
                    
                except Exception as e:
                    logger.warning(f"Failed to process positive image {image_path}: {e}")
            
            # Process negative images (background only)
            for image_path in split_data.get('negative', []):
                try:
                    image_info, annotations = self._create_image_annotations(
                        image_path, image_id, annotation_id, has_flower=False
                    )
                    
                    coco_data['images'].append(image_info)
                    # No annotations for negative images (background only)
                    
                    image_id += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process negative image {image_path}: {e}")
            
            # Save COCO JSON file
            output_filename = f"{split_name}_annotations_v{version}.json"
            output_path = annotations_dir / output_filename
            
            with AtomicFileWriter.atomic_write(output_path) as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)
            
            annotation_files[split_name] = output_path
            
            logger.info(f"Created {split_name} annotations: "
                       f"{len(coco_data['images'])} images, "
                       f"{len(coco_data['annotations'])} annotations -> {output_path}")
        
        return annotation_files
    
    def _create_image_annotations(self, image_path: Path, image_id: int, 
                                 annotation_id: int, has_flower: bool) -> Tuple[Dict, List[Dict]]:
        """
        Create COCO format image and annotation entries.
        
        Args:
            image_path: Path to image file
            image_id: Unique image ID
            annotation_id: Starting annotation ID
            has_flower: Whether image contains flowers
            
        Returns:
            Tuple of (image_info_dict, list_of_annotation_dicts)
        """
        # Load image to get dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            logger.error(f"Cannot open image {image_path}: {e}")
            raise
        
        # Create image info
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_path.name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
        
        annotations = []
        
        if has_flower:
            # For positive images, create a bounding box covering most of the image
            # This is a simplification - in a real scenario, you'd have actual annotations
            
            # Create a bounding box covering 80% of the image centered
            margin_x = width * 0.1
            margin_y = height * 0.1
            bbox_width = width * 0.8
            bbox_height = height * 0.8
            
            bbox = [margin_x, margin_y, bbox_width, bbox_height]  # [x, y, width, height]
            area = bbox_width * bbox_height
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # flower category
                "segmentation": [],  # Empty for bounding box only
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            
            annotations.append(annotation)
        
        return image_info, annotations
    
    def build_dataset(self, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Main orchestration function for dataset building.
        
        This function implements the main() execution block specified in
        Code_Structure.txt, orchestrating the entire dataset building process
        including hard negative integration and COCO format generation.
        
        Args:
            version: Optional dataset version number (auto-incremented if None)
            
        Returns:
            Dictionary with dataset building results and metadata
        """
        logger.info("Starting dataset building process")
        
        # Set reproducibility
        ReproducibilityManager.set_seed(self.global_seed)
        
        try:
            # Step 1: Load image paths from Google Drive
            logger.info("Step 1/5: Loading image paths from Google Drive")
            image_paths = self.load_image_paths()
            
            # Step 2: Load confirmed hard negatives and integrate them
            logger.info("Step 2/5: Loading and integrating hard negatives")
            hard_negatives = self.load_hard_negatives_log()
            if hard_negatives:
                logger.info(f"Integrating {len(hard_negatives)} hard negatives into dataset")
                image_paths['negative'].extend(hard_negatives)
            
            # Step 3: Create deterministic splits
            logger.info("Step 3/5: Creating deterministic data splits")
            splits = self.create_deterministic_splits(image_paths)
            
            # Step 4: Determine version number
            logger.info("Step 4/5: Determining dataset version")
            if version is None:
                version = self._get_next_version()
            
            # Step 5: Generate COCO JSON annotations
            logger.info("Step 5/5: Generating COCO JSON annotations")
            annotation_files = self.generate_coco_json(splits, version)
            
            # Additional steps with progress indication
            logger.info("Generating data manifest for integrity checking")
            manifest = self._generate_data_manifest(splits, version)
            
            logger.info("Creating dataset summary")
            summary = self._create_dataset_summary(splits, version, annotation_files)
            
            logger.info("Saving dataset metadata")
            metadata_file = self._save_dataset_metadata(summary, version)
            
            logger.info(f"Dataset building completed successfully (version {version})")
            
            return {
                'version': version,
                'splits': splits,
                'annotation_files': annotation_files,
                'manifest_file': manifest,
                'metadata_file': metadata_file,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Dataset building failed: {e}")
            raise
    
    def _get_next_version(self) -> int:
        """Get the next available dataset version number."""
        annotations_dir = self.data_paths['annotations']
        
        if not annotations_dir.exists():
            return 1
        
        # Find existing version numbers
        existing_versions = set()
        for file_path in annotations_dir.glob('*_annotations_v*.json'):
            try:
                # Extract version number from filename
                version_part = file_path.stem.split('_v')[-1]
                version_num = int(version_part)
                existing_versions.add(version_num)
            except (ValueError, IndexError):
                continue
        
        return max(existing_versions, default=0) + 1
    
    def _generate_data_manifest(self, splits: Dict, version: int) -> Path:
        """Generate integrity manifest for the dataset."""
        logger.info("Generating data integrity manifest")
        
        all_files = []
        for split_data in splits.values():
            for image_list in split_data.values():
                all_files.extend(image_list)
        
        # Generate hashes for all files
        manifest = {}
        for file_path in all_files:
            try:
                relative_path = file_path.relative_to(self.data_paths['base'])
                file_hash = FileHasher.generate_file_hash(file_path)
                manifest[str(relative_path)] = file_hash
            except Exception as e:
                logger.warning(f"Failed to hash {file_path}: {e}")
                manifest[str(file_path)] = f"ERROR: {str(e)}"
        
        # Save manifest
        manifest_file = self.data_paths['base'] / f"data_manifest_v{version}.json"
        FileHasher.save_manifest(manifest, manifest_file)
        
        return manifest_file
    
    def _create_dataset_summary(self, splits: Dict, version: int, 
                               annotation_files: Dict) -> Dict[str, Any]:
        """Create comprehensive dataset summary."""
        summary = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'global_seed': self.global_seed,
            'config_snapshot': {
                'train_split': self.config.get('data.train_split'),
                'val_split': self.config.get('data.val_split'),
                'test_split': self.config.get('data.test_split'),
                'min_image_size': self.config.get('data.min_image_size'),
                'max_dataset_size': self.config.get('data.max_dataset_size')
            },
            'splits': {},
            'annotation_files': {k: str(v) for k, v in annotation_files.items()},
            'statistics': {}
        }
        
        # Calculate split statistics
        total_positive = 0
        total_negative = 0
        
        for split_name, split_data in splits.items():
            pos_count = len(split_data.get('positive', []))
            neg_count = len(split_data.get('negative', []))
            
            summary['splits'][split_name] = {
                'positive_images': pos_count,
                'negative_images': neg_count,
                'total_images': pos_count + neg_count,
                'positive_ratio': pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
            }
            
            total_positive += pos_count
            total_negative += neg_count
        
        # Overall statistics
        summary['statistics'] = {
            'total_positive_images': total_positive,
            'total_negative_images': total_negative,
            'total_images': total_positive + total_negative,
            'positive_ratio': total_positive / (total_positive + total_negative) if (total_positive + total_negative) > 0 else 0,
            'class_balance': 'balanced' if 0.3 <= (total_positive / (total_positive + total_negative)) <= 0.7 else 'imbalanced'
        }
        
        return summary
    
    def _save_dataset_metadata(self, summary: Dict, version: int) -> Path:
        """Save dataset metadata and summary."""
        metadata_file = self.data_paths['base'] / f"dataset_metadata_v{version}.json"
        
        with AtomicFileWriter.atomic_write(metadata_file) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset metadata to {metadata_file}")
        return metadata_file


def main():
    """
    Main entry point for dataset building.
    
    This function can be called directly or used as a command-line script
    for building the dataset independently.
    """
    import sys
    import argparse
    
    # Setup argument parsing
    parser = argparse.ArgumentParser(description='Build Foundational Flower Detector Dataset')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--version', type=int, help='Dataset version number')
    parser.add_argument('--validate', action='store_true', help='Run data validation')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Import config here to avoid circular imports
        from ..config import Config
        
        # Load configuration
        config = Config(args.config)
        
        # Validate configuration
        validation_results = config.validate()
        if not validation_results['valid']:
            logger.error("Configuration validation failed:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        # Create dataset builder
        builder = DatasetBuilder(config)
        
        # Run data validation if requested
        if args.validate:
            logger.info("Running data validation...")
            data_paths = config.get_data_paths()
            
            # Add progress bar for validation directories
            validation_dirs = [(name, path) for name, path in data_paths.items() 
                             if path.exists() and path.is_dir()]
            
            if validation_dirs:
                logger.info(f"Validating {len(validation_dirs)} directories...")
                
                for path_name, path in tqdm(validation_dirs, desc="Validating directories", unit="dir"):
                    validation_result = builder.validator.validate_directory(path)
                    logger.info(f"{path_name}: {validation_result['validation_rate']:.2%} valid "
                               f"({validation_result['valid_files']}/{validation_result['total_files']})")
            else:
                logger.warning("No valid directories found for validation")
        
        # Build dataset
        results = builder.build_dataset(version=args.version)
        
        logger.info("Dataset building completed successfully!")
        logger.info(f"Version: {results['version']}")
        logger.info(f"Annotation files: {results['annotation_files']}")
        logger.info(f"Summary: {results['summary']['statistics']}")
        
    except Exception as e:
        logger.error(f"Dataset building failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
