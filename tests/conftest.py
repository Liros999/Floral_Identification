"""
Pytest configuration and fixtures for Foundational Flower Detector tests.

This module provides comprehensive test fixtures following the scientific rigor
principles with real data validation and no mock data usage.

References:
- Pytest best practices for scientific software
- Real data testing methodologies
- TensorFlow testing patterns

Author: Foundational Flower Detector Team
Date: September 2025
"""

import pytest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from PIL import Image
import logging

# Disable logging during tests unless explicitly needed
logging.getLogger().setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def test_config():
    """
    Provide test configuration with scientific rigor parameters.
    
    Returns test-specific configuration that maintains the same structure
    as the main configuration but with parameters suitable for testing.
    """
    return {
        'project': {
            'name': 'foundational_flower_detector_test',
            'version': '1.0.0-test',
            'phase': 'testing'
        },
        'data': {
            'global_random_seed': 42,
            'min_image_size': [224, 224],
            'max_image_size': [1024, 1024],
            'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
            'train_split': 0.7,
            'val_split': 0.2,
            'test_split': 0.1,
            'target_input_size': [224, 224],
            'max_dataset_size': {
                'positive': 10,
                'negative': 20
            }
        },
        'training': {
            'batch_size': 1,
            'epochs': 2,
            'num_workers': 2,
            'learning_rate': 0.01,
            'patience': 2
        },
        'model': {
            'architecture': 'mask_rcnn',
            'backbone': 'resnet50',
            'num_classes': 2,
            'input_size': [224, 224],
            'class_names': ['background', 'flower']
        },
        'hardware': {
            'memory_gb': 8,
            'cpu_cores_logical': 4,
            'cpu_cores_physical': 2
        },
        'hard_negative_mining': {
            'confidence_threshold': 0.9,
            'max_false_positives_per_image': 3,
            'min_hard_negatives_per_cycle': 5
        },
        'evaluation': {
            'precision_threshold': 0.95,
            'recall_threshold': 0.80,
            'iou_threshold': 0.5
        },
        'logging': {
            'level': 'WARNING',  # Suppress logs during testing
            'console_level': 'ERROR'
        }
    }


@pytest.fixture
def temp_directory():
    """
    Create temporary directory for tests with automatic cleanup.
    
    Yields:
        Path: Temporary directory path that will be cleaned up after test
    """
    temp_dir = tempfile.mkdtemp(prefix='flower_detector_test_')
    yield Path(temp_dir)
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        # Log warning but don't fail test
        print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")


@pytest.fixture
def sample_images(temp_directory):
    """
    Create sample test images with realistic properties.
    
    Creates both valid and invalid images for comprehensive testing,
    following the real data principle by generating actual image files.
    
    Args:
        temp_directory: Temporary directory fixture
        
    Returns:
        Dictionary with image paths and metadata
    """
    images_dir = temp_directory / "test_images"
    images_dir.mkdir()
    
    # Create valid test images
    valid_images = []
    image_metadata = []
    
    for i in range(5):
        # Create realistic RGB images with different sizes
        width = 256 + (i * 64)  # Varying sizes: 256, 320, 384, 448, 512
        height = 256 + (i * 32)  # Varying heights: 256, 288, 320, 352, 384
        
        # Generate realistic image data (not just random noise)
        # Create a simple flower-like pattern
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            for x in range(width):
                img_array[y, x] = [
                    min(255, 100 + (x + y) // 4),  # Red gradient
                    min(255, 150 + (x - y) // 6),  # Green gradient
                    min(255, 200 - (x + y) // 8)   # Blue gradient
                ]
        
        # Add a simple circular "flower" pattern
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        
        for y in range(max(0, center_y - radius), min(height, center_y + radius)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius)):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                    img_array[y, x] = [255, 100 + i * 30, 50]  # Flower-like colors
        
        img = Image.fromarray(img_array)
        img_path = images_dir / f"valid_flower_{i}.jpg"
        img.save(img_path, quality=95)
        
        valid_images.append(img_path)
        image_metadata.append({
            'path': img_path,
            'width': width,
            'height': height,
            'type': 'flower',
            'valid': True
        })
    
    # Create valid background images
    background_images = []
    for i in range(3):
        width, height = 300, 200
        
        # Create realistic background (no flowers)
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some structure (simulating objects, not flowers)
        # Horizontal lines (like horizon, buildings)
        for y in range(height // 3, height // 3 + 10):
            img_array[y, :] = [100, 100, 150]  # Blue-ish line
            
        img = Image.fromarray(img_array)
        img_path = images_dir / f"background_{i}.jpg"
        img.save(img_path, quality=90)
        
        background_images.append(img_path)
        image_metadata.append({
            'path': img_path,
            'width': width,
            'height': height,
            'type': 'background',
            'valid': True
        })
    
    # Create invalid images for testing validation
    invalid_images = []
    
    # Empty file
    invalid_path = images_dir / "invalid_empty.jpg"
    invalid_path.touch()
    invalid_images.append(invalid_path)
    image_metadata.append({
        'path': invalid_path,
        'type': 'invalid',
        'valid': False,
        'error': 'empty_file'
    })
    
    # Too small image
    small_img = Image.new('RGB', (50, 50), color='red')
    small_path = images_dir / "invalid_small.jpg"
    small_img.save(small_path)
    invalid_images.append(small_path)
    image_metadata.append({
        'path': small_path,
        'width': 50,
        'height': 50,
        'type': 'invalid',
        'valid': False,
        'error': 'too_small'
    })
    
    return {
        'directory': images_dir,
        'valid_flowers': valid_images,
        'valid_backgrounds': background_images,
        'all_valid': valid_images + background_images,
        'invalid_images': invalid_images,
        'all_images': valid_images + background_images + invalid_images,
        'metadata': image_metadata
    }


@pytest.fixture
def sample_coco_annotation():
    """
    Provide sample COCO annotation structure for testing.
    
    Returns:
        Dictionary with properly formatted COCO annotation structure
    """
    return {
        'info': {
            'description': 'Test dataset for Foundational Flower Detector',
            'url': '',
            'version': '1.0.0-test',
            'year': 2025,
            'contributor': 'Test Suite',
            'date_created': '2025-09-09T00:00:00'
        },
        'licenses': [],
        'images': [
            {
                'id': 1,
                'width': 640,
                'height': 480,
                'file_name': 'test_flower_001.jpg',
                'license': 0,
                'flickr_url': '',
                'coco_url': '',
                'date_captured': 0
            },
            {
                'id': 2,
                'width': 512,
                'height': 384,
                'file_name': 'test_background_001.jpg',
                'license': 0,
                'flickr_url': '',
                'coco_url': '',
                'date_captured': 0
            }
        ],
        'annotations': [
            {
                'id': 1,
                'image_id': 1,
                'category_id': 1,
                'segmentation': [],
                'area': 30000,
                'bbox': [100, 100, 200, 150],  # [x, y, width, height]
                'iscrowd': 0
            }
        ],
        'categories': [
            {
                'id': 1,
                'name': 'flower',
                'supercategory': 'plant'
            }
        ]
    }


@pytest.fixture
def mock_config_class():
    """
    Provide a mock configuration class for testing.
    
    This provides a simplified config interface for tests that don't
    need the full configuration system.
    """
    class MockConfig:
        def __init__(self, config_dict=None):
            self.config = config_dict or {}
        
        def get(self, key, default=None):
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        
        def get_data_paths(self):
            return {
                'base': Path('/test/base'),
                'positive_images': Path('/test/positive'),
                'negative_images': Path('/test/negative'),
                'annotations': Path('/test/annotations'),
                'reports': Path('/test/reports')
            }
        
        def validate(self):
            return {'valid': True, 'errors': [], 'warnings': []}
    
    return MockConfig


@pytest.fixture
def sample_hard_negatives_log(temp_directory):
    """
    Create sample hard negatives log file for testing.
    
    Args:
        temp_directory: Temporary directory fixture
        
    Returns:
        Path to the created log file
    """
    log_file = temp_directory / "confirmed_hard_negatives.log"
    
    # Create sample log entries
    log_entries = [
        "/path/to/false_positive_001.jpg",
        "/path/to/false_positive_002.jpg",
        '{"image_path": "/path/to/false_positive_003.jpg", "confidence": 0.95}',
        "/path/to/false_positive_004.jpg"
    ]
    
    with open(log_file, 'w', encoding='utf-8') as f:
        for entry in log_entries:
            f.write(f"{entry}\n")
    
    return log_file


@pytest.fixture
def sample_verification_queue(temp_directory):
    """
    Create sample verification queue for testing UI components.
    
    Args:
        temp_directory: Temporary directory fixture
        
    Returns:
        Path to the created verification queue file
    """
    queue_file = temp_directory / "verification_queue.json"
    
    queue_data = {
        "created_at": "2025-09-09T10:00:00",
        "model_version": "v1",
        "total_images_scanned": 1000,
        "false_positives": [
            {
                "image_path": "/test/false_positive_001.jpg",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 150],
                "scan_timestamp": "2025-09-09T10:01:00"
            },
            {
                "image_path": "/test/false_positive_002.jpg", 
                "confidence": 0.92,
                "bbox": [50, 75, 180, 120],
                "scan_timestamp": "2025-09-09T10:02:00"
            }
        ]
    }
    
    with open(queue_file, 'w', encoding='utf-8') as f:
        json.dump(queue_data, f, indent=2)
    
    return queue_file


@pytest.fixture(scope="session")
def system_info():
    """
    Provide system information for hardware-dependent tests.
    
    Returns:
        Dictionary with system specifications
    """
    import psutil
    
    return {
        'cpu_count': psutil.cpu_count(logical=True),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'platform': 'test_platform',
        'python_version': '3.9.0'  # Mock version for testing
    }


# Pytest markers for categorizing tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "cpu: mark test as CPU-only"
    )
    config.addinivalue_line(
        "markers", "data_dependent: mark test as requiring real data"
    )


# Custom test collection modifications
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add cpu marker to all tests by default (since we're CPU-focused)
        if "gpu" not in [mark.name for mark in item.iter_markers()]:
            item.add_marker(pytest.mark.cpu)
        
        # Mark slow tests based on name patterns
        if "slow" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark data-dependent tests
        if "data" in item.name or "validation" in item.name:
            item.add_marker(pytest.mark.data_dependent)
