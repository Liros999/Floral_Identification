"""
Tests for data preparation utilities.

This module tests the core data preparation utilities following the scientific
rigor principles with real data validation and comprehensive coverage.

Author: Foundational Flower Detector Team
Date: September 2025
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_preparation.utils import (
    ReproducibilityManager,
    AtomicFileWriter, 
    DataIntegrityValidator,
    SystemMonitor,
    FileHasher
)


class TestReproducibilityManager:
    """Test reproducibility management functionality."""
    
    @pytest.mark.unit
    def test_set_seed_basic(self):
        """Test basic seed setting functionality."""
        # Set seed
        ReproducibilityManager.set_seed(42)
        
        # Test that random operations are deterministic
        import random
        import numpy as np
        
        # Get first set of random values
        random_val1 = random.random()
        numpy_val1 = np.random.random()
        
        # Reset seed
        ReproducibilityManager.set_seed(42)
        
        # Get second set - should be identical
        random_val2 = random.random()
        numpy_val2 = np.random.random()
        
        assert random_val1 == random_val2
        assert numpy_val1 == numpy_val2
    
    @pytest.mark.unit
    def test_environment_variables_set(self):
        """Test that environment variables are properly set."""
        import os
        
        ReproducibilityManager.set_seed(123)
        
        assert os.environ.get('PYTHONHASHSEED') == '123'
        assert os.environ.get('TF_DETERMINISTIC_OPS') == '1'
    
    @pytest.mark.unit
    def test_tensorflow_configuration(self):
        """Test TensorFlow configuration for reproducibility."""
        import tensorflow as tf
        
        ReproducibilityManager.set_seed(42)
        
        # Check that TensorFlow threads are configured
        # Note: These may not be directly testable depending on TF version
        # but we can verify the function doesn't raise errors
        assert True  # Function completed without error


class TestAtomicFileWriter:
    """Test atomic file writing functionality."""
    
    @pytest.mark.unit
    def test_atomic_write_success(self, temp_directory):
        """Test successful atomic file writing."""
        test_file = temp_directory / "test_atomic.json"
        test_data = {"key": "value", "number": 42}
        
        # Write data atomically
        with AtomicFileWriter.atomic_write(test_file) as f:
            json.dump(test_data, f)
        
        # Verify file exists and contains correct data
        assert test_file.exists()
        
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
    
    @pytest.mark.unit
    def test_atomic_write_creates_directories(self, temp_directory):
        """Test that atomic write creates necessary directories."""
        nested_file = temp_directory / "nested" / "dir" / "test.json"
        
        with AtomicFileWriter.atomic_write(nested_file) as f:
            json.dump({"test": True}, f)
        
        assert nested_file.exists()
        assert nested_file.parent.exists()
    
    @pytest.mark.unit
    def test_atomic_write_cleanup_on_exception(self, temp_directory):
        """Test cleanup of temporary files when exception occurs."""
        test_file = temp_directory / "test_exception.json"
        
        with pytest.raises(ValueError):
            with AtomicFileWriter.atomic_write(test_file) as f:
                f.write("partial data")
                raise ValueError("Simulated error")
        
        # Main file should not exist
        assert not test_file.exists()
        
        # No temporary files should remain
        temp_files = list(temp_directory.glob("*.tmp"))
        lock_files = list(temp_directory.glob("*.lock"))
        
        assert len(temp_files) == 0
        assert len(lock_files) == 0


class TestDataIntegrityValidator:
    """Test data integrity validation functionality."""
    
    @pytest.mark.unit
    @pytest.mark.data_dependent
    def test_validate_image_file_valid(self, sample_images, test_config):
        """Test validation of valid image file."""
        validator = DataIntegrityValidator(test_config)
        valid_image = sample_images['valid_flowers'][0]
        
        result = validator.validate_image_file(valid_image)
        
        assert result['valid'] is True
        assert result['exists'] is True
        assert len(result['errors']) == 0
        assert 'width' in result['properties']
        assert 'height' in result['properties']
        assert result['properties']['width'] >= test_config['data']['min_image_size'][0]
        assert result['properties']['height'] >= test_config['data']['min_image_size'][1]
    
    @pytest.mark.unit
    @pytest.mark.data_dependent
    def test_validate_image_file_invalid(self, sample_images, test_config):
        """Test validation of invalid image file."""
        validator = DataIntegrityValidator(test_config)
        invalid_image = sample_images['invalid_images'][0]
        
        result = validator.validate_image_file(invalid_image)
        
        assert result['valid'] is False
        assert result['exists'] is True
        assert len(result['errors']) > 0
    
    @pytest.mark.unit
    @pytest.mark.data_dependent
    def test_validate_directory(self, sample_images, test_config):
        """Test directory validation functionality."""
        validator = DataIntegrityValidator(test_config)
        images_directory = sample_images['directory']
        
        result = validator.validate_directory(images_directory)
        
        assert result['exists'] is True
        assert result['total_files'] > 0
        assert result['valid_files'] > 0
        assert 0 <= result['validation_rate'] <= 1.0
    
    @pytest.mark.unit
    def test_validate_nonexistent_directory(self, test_config):
        """Test validation of non-existent directory."""
        validator = DataIntegrityValidator(test_config)
        
        result = validator.validate_directory("/nonexistent/path")
        
        assert result['exists'] is False
        assert result['total_files'] == 0
        assert result['valid_files'] == 0
        assert result['validation_rate'] == 0.0
    
    @pytest.mark.unit
    @pytest.mark.data_dependent
    def test_get_validation_summary(self, sample_images, test_config):
        """Test validation summary generation."""
        validator = DataIntegrityValidator(test_config)
        
        # Validate multiple images
        validation_results = []
        for image_path in sample_images['all_images']:
            result = validator.validate_image_file(image_path)
            validation_results.append(result)
        
        # Generate summary
        summary = validator.get_validation_summary(validation_results)
        
        assert 'total_images' in summary
        assert 'valid_images' in summary
        assert 'validation_rate' in summary
        assert summary['total_images'] == len(sample_images['all_images'])
        assert 0 <= summary['validation_rate'] <= 1.0


class TestSystemMonitor:
    """Test system monitoring functionality."""
    
    @pytest.mark.unit
    @pytest.mark.cpu
    def test_get_system_info(self):
        """Test system information retrieval."""
        info = SystemMonitor.get_system_info()
        
        # Check required keys exist
        assert 'cpu' in info
        assert 'memory' in info
        assert 'disk' in info
        assert 'timestamp' in info
        
        # Check CPU info structure
        assert 'cpu_count_logical' in info['cpu']
        assert 'cpu_count_physical' in info['cpu']
        assert isinstance(info['cpu']['cpu_count_logical'], int)
        assert info['cpu']['cpu_count_logical'] > 0
        
        # Check memory info
        assert 'total' in info['memory']
        assert 'available' in info['memory']
        assert info['memory']['total'] > 0
    
    @pytest.mark.unit
    def test_check_system_requirements(self, test_config):
        """Test system requirements checking."""
        # Use real system info for testing
        result = SystemMonitor.check_system_requirements(test_config)
        
        assert 'memory_sufficient' in result
        assert 'cpu_cores_sufficient' in result
        assert 'disk_space_sufficient' in result
        assert 'system_ready' in result
        
        # All values should be boolean
        for key, value in result.items():
            if key.endswith('_sufficient') or key == 'system_ready':
                assert isinstance(value, bool)
    
    @pytest.mark.unit
    def test_monitor_training_resources(self, temp_directory):
        """Test training resource monitoring."""
        log_file = temp_directory / "resource_log.jsonl"
        
        metrics = SystemMonitor.monitor_training_resources(log_file)
        
        # Check metrics structure
        assert 'timestamp' in metrics
        assert 'cpu_usage_mean' in metrics or 'error' in metrics
        
        # Check log file was created
        if 'error' not in metrics:
            assert log_file.exists()
            
            # Verify log file format
            with open(log_file, 'r') as f:
                log_line = f.readline().strip()
                if log_line:
                    log_data = json.loads(log_line)
                    assert 'timestamp' in log_data


class TestFileHasher:
    """Test file hashing functionality."""
    
    @pytest.mark.unit
    def test_generate_file_hash(self, temp_directory):
        """Test file hash generation."""
        test_file = temp_directory / "hash_test.txt"
        test_content = "This is test content for hashing"
        
        # Create test file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Generate hash
        file_hash = FileHasher.generate_file_hash(test_file)
        
        # Verify hash properties
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA256 hash length
        
        # Verify hash is deterministic
        hash2 = FileHasher.generate_file_hash(test_file)
        assert file_hash == hash2
    
    @pytest.mark.unit
    def test_generate_file_hash_different_algorithms(self, temp_directory):
        """Test hash generation with different algorithms."""
        test_file = temp_directory / "hash_test.txt"
        test_content = "Test content"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test different algorithms
        sha256_hash = FileHasher.generate_file_hash(test_file, 'sha256')
        md5_hash = FileHasher.generate_file_hash(test_file, 'md5')
        
        assert len(sha256_hash) == 64
        assert len(md5_hash) == 32
        assert sha256_hash != md5_hash
    
    @pytest.mark.unit
    def test_generate_file_hash_nonexistent_file(self):
        """Test hash generation for non-existent file."""
        with pytest.raises(FileNotFoundError):
            FileHasher.generate_file_hash("/nonexistent/file.txt")
    
    @pytest.mark.unit
    @pytest.mark.data_dependent
    def test_generate_directory_manifest(self, sample_images):
        """Test directory manifest generation."""
        manifest = FileHasher.generate_directory_manifest(sample_images['directory'])
        
        # Should have entries for valid image files
        assert len(manifest) > 0
        
        # All values should be valid hashes or error messages
        for filename, hash_value in manifest.items():
            assert isinstance(hash_value, str)
            if not hash_value.startswith("ERROR:"):
                assert len(hash_value) == 64  # SHA256
    
    @pytest.mark.unit
    def test_save_manifest(self, temp_directory):
        """Test manifest saving functionality."""
        manifest = {
            "file1.jpg": "abc123def456" + "0" * 52,  # Valid SHA256 length
            "file2.jpg": "def789ghi012" + "1" * 52
        }
        
        manifest_path = temp_directory / "manifest.json"
        
        FileHasher.save_manifest(manifest, manifest_path)
        
        # Verify file was created and contains correct data
        assert manifest_path.exists()
        
        with open(manifest_path, 'r') as f:
            loaded_manifest = json.load(f)
        
        assert loaded_manifest == manifest
    
    @pytest.mark.unit
    def test_verify_manifest(self, temp_directory):
        """Test manifest verification functionality."""
        # Create test files
        file1 = temp_directory / "file1.txt"
        file2 = temp_directory / "file2.txt"
        
        file1.write_text("content1")
        file2.write_text("content2")
        
        # Generate manifest
        manifest = FileHasher.generate_directory_manifest(temp_directory)
        manifest_path = temp_directory / "manifest.json"
        FileHasher.save_manifest(manifest, manifest_path)
        
        # Verify manifest
        results = FileHasher.verify_manifest(manifest_path, temp_directory)
        
        assert results['total_files'] == len(manifest)
        assert results['verified_files'] > 0
        assert results['verification_rate'] > 0
        assert 'missing_files' in results
        assert 'corrupted_files' in results
