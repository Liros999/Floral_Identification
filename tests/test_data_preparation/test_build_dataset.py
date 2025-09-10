"""
Tests for dataset building functionality.

This module tests the COCO dataset construction and management capabilities
following scientific rigor principles with real data validation.

Author: Foundational Flower Detector Team
Date: September 2025
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_preparation.build_dataset import DatasetBuilder


class TestDatasetBuilder:
    """Test dataset building functionality."""
    
    @pytest.fixture
    def dataset_builder(self, mock_config_class):
        """Create dataset builder with mock config."""
        config = mock_config_class({
            'data': {
                'global_random_seed': 42,
                'train_split': 0.7,
                'val_split': 0.2,
                'test_split': 0.1,
                'image_extensions': ['.jpg', '.jpeg', '.png'],
                'positive_images_subpath': 'positive_images',
                'negative_images_subpath': 'negative_images',
                'annotations_subpath': 'annotations',
                'metadata_subpath': 'metadata'
            }
        })
        return DatasetBuilder(config)
    
    @pytest.mark.unit
    def test_init(self, dataset_builder):
        """Test dataset builder initialization."""
        assert dataset_builder.config is not None
        assert dataset_builder.global_seed == 42
        assert dataset_builder.validator is not None
    
    @pytest.mark.unit
    @pytest.mark.data_dependent  
    def test_load_images_from_directory(self, dataset_builder, sample_images):
        """Test loading images from directory."""
        images = dataset_builder._load_images_from_directory(sample_images['directory'])
        
        # Should find valid images
        assert len(images) > 0
        
        # All returned paths should exist and be valid
        for image_path in images:
            assert image_path.exists()
            assert image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']
    
    @pytest.mark.unit
    def test_create_deterministic_splits(self, dataset_builder, sample_images):
        """Test deterministic data splitting."""
        image_paths = {
            'positive': sample_images['valid_flowers'],
            'negative': sample_images['valid_backgrounds']
        }
        
        # Create splits
        splits = dataset_builder.create_deterministic_splits(image_paths)
        
        # Check structure
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        for split_name in ['train', 'val', 'test']:
            assert 'positive' in splits[split_name]
            assert 'negative' in splits[split_name]
        
        # Check that splits are deterministic (run twice)
        splits2 = dataset_builder.create_deterministic_splits(image_paths)
        
        # Should be identical
        for split_name in ['train', 'val', 'test']:
            for label in ['positive', 'negative']:
                assert splits[split_name][label] == splits2[split_name][label]
    
    @pytest.mark.unit
    def test_create_deterministic_splits_ratios(self, dataset_builder, sample_images):
        """Test that split ratios are approximately correct."""
        # Create larger sample for better ratio testing
        image_paths = {
            'positive': sample_images['valid_flowers'] * 10,  # Duplicate to get more samples
            'negative': sample_images['valid_backgrounds'] * 10
        }
        
        splits = dataset_builder.create_deterministic_splits(image_paths)
        
        # Count total images
        total_positive = sum(len(splits[split]['positive']) for split in splits)
        total_negative = sum(len(splits[split]['negative']) for split in splits)
        
        # Check ratios (allow some tolerance due to rounding)
        train_ratio = len(splits['train']['positive']) / total_positive
        val_ratio = len(splits['val']['positive']) / total_positive
        test_ratio = len(splits['test']['positive']) / total_positive
        
        assert abs(train_ratio - 0.7) < 0.1
        assert abs(val_ratio - 0.2) < 0.1
        assert abs(test_ratio - 0.1) < 0.1
    
    @pytest.mark.unit
    def test_load_hard_negatives_log_nonexistent(self, dataset_builder):
        """Test loading hard negatives when log doesn't exist."""
        hard_negatives = dataset_builder.load_hard_negatives_log()
        
        # Should return empty list when file doesn't exist
        assert hard_negatives == []
    
    @pytest.mark.unit
    def test_load_hard_negatives_log_existing(self, dataset_builder, sample_hard_negatives_log):
        """Test loading hard negatives from existing log."""
        # Mock the paths to point to our test log
        dataset_builder.data_paths['base'] = sample_hard_negatives_log.parent
        
        hard_negatives = dataset_builder.load_hard_negatives_log()
        
        # Should load entries from log file
        assert len(hard_negatives) > 0
        
        # All entries should be Path objects
        for hn in hard_negatives:
            assert isinstance(hn, Path)
    
    @pytest.mark.unit
    def test_create_image_annotations_positive(self, dataset_builder, sample_images):
        """Test creating annotations for positive (flower) images."""
        image_path = sample_images['valid_flowers'][0]
        
        image_info, annotations = dataset_builder._create_image_annotations(
            image_path, image_id=1, annotation_id=1, has_flower=True
        )
        
        # Check image info structure
        assert image_info['id'] == 1
        assert image_info['file_name'] == image_path.name
        assert 'width' in image_info
        assert 'height' in image_info
        
        # Should have annotations for flower
        assert len(annotations) > 0
        
        # Check annotation structure
        annotation = annotations[0]
        assert annotation['id'] == 1
        assert annotation['image_id'] == 1
        assert annotation['category_id'] == 1  # Flower category
        assert 'bbox' in annotation
        assert 'area' in annotation
    
    @pytest.mark.unit
    def test_create_image_annotations_negative(self, dataset_builder, sample_images):
        """Test creating annotations for negative (background) images."""
        image_path = sample_images['valid_backgrounds'][0]
        
        image_info, annotations = dataset_builder._create_image_annotations(
            image_path, image_id=2, annotation_id=10, has_flower=False
        )
        
        # Check image info structure
        assert image_info['id'] == 2
        assert image_info['file_name'] == image_path.name
        
        # Should have no annotations for background
        assert len(annotations) == 0
    
    @pytest.mark.unit
    def test_generate_coco_json(self, dataset_builder, sample_images, temp_directory):
        """Test COCO JSON generation."""
        # Setup test splits
        splits = {
            'train': {
                'positive': sample_images['valid_flowers'][:2],
                'negative': sample_images['valid_backgrounds'][:2]
            },
            'val': {
                'positive': sample_images['valid_flowers'][2:3],
                'negative': sample_images['valid_backgrounds'][2:3] if len(sample_images['valid_backgrounds']) > 2 else []
            }
        }
        
        # Mock annotations directory
        dataset_builder.data_paths['annotations'] = temp_directory
        
        # Generate COCO JSON
        annotation_files = dataset_builder.generate_coco_json(splits, version=1)
        
        # Check that files were created
        assert 'train' in annotation_files
        assert 'val' in annotation_files
        
        for split_name, file_path in annotation_files.items():
            assert Path(file_path).exists()
            
            # Load and verify COCO format
            with open(file_path, 'r') as f:
                coco_data = json.load(f)
            
            # Check COCO structure
            assert 'info' in coco_data
            assert 'images' in coco_data
            assert 'annotations' in coco_data
            assert 'categories' in coco_data
            
            # Should have images from both positive and negative sets
            expected_image_count = len(splits[split_name]['positive']) + len(splits[split_name]['negative'])
            assert len(coco_data['images']) == expected_image_count
            
            # Categories should include flower
            assert len(coco_data['categories']) == 1
            assert coco_data['categories'][0]['name'] == 'flower'
    
    @pytest.mark.unit
    def test_get_next_version(self, dataset_builder, temp_directory):
        """Test version number generation."""
        # Mock annotations directory
        dataset_builder.data_paths['annotations'] = temp_directory
        
        # Should start at version 1 with no existing files
        version = dataset_builder._get_next_version()
        assert version == 1
        
        # Create some existing version files
        (temp_directory / "train_annotations_v1.json").touch()
        (temp_directory / "train_annotations_v3.json").touch()
        
        # Should return next available version
        version = dataset_builder._get_next_version()
        assert version == 4
    
    @pytest.mark.unit
    @pytest.mark.data_dependent
    def test_create_dataset_summary(self, dataset_builder, sample_images):
        """Test dataset summary creation."""
        splits = {
            'train': {
                'positive': sample_images['valid_flowers'][:3],
                'negative': sample_images['valid_backgrounds'][:2]
            },
            'val': {
                'positive': sample_images['valid_flowers'][3:4],
                'negative': sample_images['valid_backgrounds'][2:3] if len(sample_images['valid_backgrounds']) > 2 else []
            }
        }
        
        summary = dataset_builder._create_dataset_summary(splits, version=1, annotation_files={})
        
        # Check summary structure
        assert 'version' in summary
        assert 'created_at' in summary
        assert 'global_seed' in summary
        assert 'splits' in summary
        assert 'statistics' in summary
        
        # Check split statistics
        for split_name in ['train', 'val']:
            assert split_name in summary['splits']
            split_stats = summary['splits'][split_name]
            assert 'positive_images' in split_stats
            assert 'negative_images' in split_stats
            assert 'total_images' in split_stats
        
        # Check overall statistics
        stats = summary['statistics']
        assert 'total_positive_images' in stats
        assert 'total_negative_images' in stats
        assert 'total_images' in stats
        assert 'positive_ratio' in stats
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.data_dependent
    def test_build_dataset_integration(self, dataset_builder, sample_images, temp_directory):
        """Integration test for complete dataset building process."""
        # Mock the image loading to use our sample images
        def mock_load_image_paths():
            return {
                'positive': sample_images['valid_flowers'],
                'negative': sample_images['valid_backgrounds']
            }
        
        # Mock paths to use temp directory
        dataset_builder.data_paths['annotations'] = temp_directory
        dataset_builder.data_paths['base'] = temp_directory
        dataset_builder.load_image_paths = mock_load_image_paths
        
        # Build dataset
        results = dataset_builder.build_dataset(version=1)
        
        # Check results structure
        assert 'version' in results
        assert 'splits' in results
        assert 'annotation_files' in results
        assert 'summary' in results
        
        # Check that annotation files were created
        for split_name in ['train', 'val', 'test']:
            assert split_name in results['annotation_files']
            annotation_file = Path(results['annotation_files'][split_name])
            assert annotation_file.exists()
            
            # Verify COCO format
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
            assert 'images' in coco_data
            assert 'annotations' in coco_data
            assert 'categories' in coco_data
