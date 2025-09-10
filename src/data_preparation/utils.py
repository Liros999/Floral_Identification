"""
Data preparation utilities for Foundational Flower Detector.

This module implements the core utilities for scientific data handling as specified
in the Code_Structure.txt documentation. All utilities follow the scientific rigor
principles with real data validation, atomic operations, and reproducibility.

Key Features:
- Reproducibility management (Decision A3 from architecture)
- Atomic file operations (Decision A1 from architecture)
- Data integrity validation with real data enforcement
- System monitoring for Intel Core Ultra 7 optimization
- File hashing for data integrity tracking

References:
- Code_Structure.txt: Detailed functionality specifications
- Scientific computing best practices for reproducibility
- Intel CPU optimization strategies

Author: Foundational Flower Detector Team
Date: September 2025
"""

import os
import json
import random
import hashlib
import tempfile
import shutil
import logging
try:
    import fcntl
except ImportError:
    # fcntl is not available on Windows
    fcntl = None
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Any, Union, List, Dict, Tuple, Optional
from contextlib import contextmanager
import psutil  # For system monitoring
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """
    Ensures complete reproducibility across all stochastic processes.
    
    Implementation of Decision A3 from architecture documents. This class provides
    deterministic behavior for all random operations in the pipeline, which is
    critical for scientific reproducibility and experiment validation.
    
    Methods follow the specifications in Code_Structure.txt with explicit
    seed setting for all relevant libraries and TensorFlow deterministic operations.
    """
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set random seed for all relevant libraries to ensure reproducibility.
        
        This function implements the cornerstone of our reproducibility strategy
        as specified in the architecture documents. It ensures that any stochastic
        process (data shuffling, model weight initialization, augmentation) is
        deterministic across runs.
        
        Args:
            seed: Random seed value (default: 42 as specified in config)
        
        Note:
            This function must be called before any data loading or model
            initialization to ensure complete reproducibility.
        """
        logger.info(f"Setting global random seed to {seed} for reproducibility")
        
        # Python random module
        random.seed(seed)
        
        # NumPy random state
        np.random.seed(seed)
        
        # TensorFlow random operations
        tf.random.set_seed(seed)
        
        # Environment variables for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Even though CPU-only
        
        # Configure TensorFlow for CPU determinism
        # Following Intel CPU optimization guidelines
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        # Disable GPU if available (CPU-only training)
        tf.config.set_visible_devices([], 'GPU')
        
        logger.info("Reproducibility settings applied successfully")
        logger.debug(f"Environment variables set: PYTHONHASHSEED={seed}, TF_DETERMINISTIC_OPS=1")


class AtomicFileWriter:
    """
    Thread-safe file writing to prevent race conditions.
    
    Implementation of Decision A1 from architecture documents. This class provides
    atomic file operations critical for our JSON-based state management in the
    hard negative mining cycle.
    
    The implementation uses file locking mechanisms to prevent corruption during
    concurrent access, which is essential for the asynchronous human-in-the-loop
    workflow described in the architecture.
    """
    
    @staticmethod
    @contextmanager
    def atomic_write(filepath: Union[str, Path], mode: str = 'w', encoding: str = 'utf-8'):
        """
        Context manager for atomic file writing with file locking.
        
        This method implements the file-locking mechanism specified in
        Code_Structure.txt. It creates a temporary .lock file, writes data
        to the target file, and then deletes the .lock file. Any other process
        attempting to write will wait until the lock is released.
        
        Args:
            filepath: Target file path
            mode: File open mode (default: 'w')
            encoding: File encoding (default: 'utf-8')
            
        Yields:
            File handle for writing
            
        Raises:
            IOError: If lock cannot be acquired
            Exception: For other file operation errors
            
        Example:
            >>> with AtomicFileWriter.atomic_write('data.json') as f:
            ...     json.dump(data, f)
        """
        filepath = Path(filepath)
        lock_path = filepath.with_suffix(filepath.suffix + '.lock')
        temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create and acquire lock
            with open(lock_path, 'w', encoding=encoding) as lock_file:
                try:
                    # File locking (Unix) or simple existence check (Windows)
                    if fcntl is not None:
                        # Unix/Linux: Use fcntl for proper file locking
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        logger.debug(f"Acquired file lock for {filepath}")
                    else:
                        # Windows: Use simple file existence as lock mechanism
                        logger.debug(f"Using file existence lock for {filepath}")
                    
                    # Write to temporary file
                    with open(temp_path, mode, encoding=encoding) as temp_file:
                        yield temp_file
                    
                    # Atomic move to final location
                    shutil.move(str(temp_path), str(filepath))
                    logger.debug(f"Atomically wrote {filepath}")
                    
                except IOError as e:
                    logger.error(f"Could not acquire lock for {filepath}: {e}")
                    raise IOError(f"File lock acquisition failed: {e}")
                    
                finally:
                    # Release lock
                    if fcntl is not None:
                        try:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        except Exception as e:
                            logger.warning(f"Failed to release lock: {e}")
                        
        except Exception as e:
            # Cleanup temporary file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")
            
            logger.error(f"Atomic write failed for {filepath}: {e}")
            raise
            
        finally:
            # Cleanup lock file
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup lock file {lock_path}: {e}")


class DataIntegrityValidator:
    """
    Comprehensive data validation and integrity checking.
    
    This class implements rigorous data validation following the "NEVER USE FAKE DATA"
    principle from the project rules. It provides comprehensive validation for image
    files, ensuring they meet the scientific standards required for the project.
    
    All validation follows real data enforcement principles with no mock or synthetic
    data permitted in the pipeline.
    """
    
    def __init__(self, config):
        """
        Initialize data integrity validator.
        
        Args:
            config: Configuration object containing validation parameters
        """
        self.config = config
        self.min_image_size = config.get('data.min_image_size', [224, 224])
        self.max_image_size = config.get('data.max_image_size', [1024, 1024])
        self.valid_extensions = config.get('data.image_extensions', ['.jpg', '.jpeg', '.png', '.bmp'])
        self.target_input_size = config.get('data.target_input_size', [224, 224])
        
        logger.info(f"Initialized DataIntegrityValidator with min_size={self.min_image_size}, "
                   f"max_size={self.max_image_size}, extensions={self.valid_extensions}")
    
    def validate_image_file(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a single image file for integrity and basic properties.
        
        Performs comprehensive validation including file existence, format validation,
        size constraints, and image integrity checks. This ensures only real,
        properly formatted images enter the training pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with validation results containing:
            - path: File path
            - exists: Whether file exists
            - valid: Overall validation status
            - errors: List of validation errors
            - properties: Image properties (if valid)
        """
        image_path = Path(image_path)
        
        validation_result = {
            'path': str(image_path),
            'exists': image_path.exists(),
            'valid': False,
            'errors': [],
            'warnings': [],
            'properties': {}
        }
        
        if not validation_result['exists']:
            validation_result['errors'].append('File does not exist')
            return validation_result
        
        try:
            # Check file extension
            file_extension = image_path.suffix.lower()
            if file_extension not in self.valid_extensions:
                validation_result['errors'].append(
                    f'Invalid extension: {file_extension}. '
                    f'Allowed: {", ".join(self.valid_extensions)}'
                )
            
            # Check file size (basic sanity check)
            file_size = image_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                validation_result['errors'].append(f'File too small: {file_size} bytes')
            elif file_size > 50 * 1024 * 1024:  # Larger than 50MB is suspicious
                validation_result['warnings'].append(f'Large file size: {file_size / (1024*1024):.1f} MB')
            
            # Validate image with PIL (primary validation)
            with Image.open(image_path) as img:
                # Verify image integrity
                img.verify()
                
            # Reopen to get properties (verify closes the image)
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                validation_result['properties'] = {
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'format': format_name,
                    'file_size': file_size,
                    'aspect_ratio': width / height if height > 0 else 0
                }
                
                # Check minimum dimensions
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    validation_result['errors'].append(
                        f'Image too small: {width}x{height}, '
                        f'minimum: {self.min_image_size[0]}x{self.min_image_size[1]}'
                    )
                
                # Check maximum dimensions
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    validation_result['warnings'].append(
                        f'Image large: {width}x{height}, '
                        f'maximum recommended: {self.max_image_size[0]}x{self.max_image_size[1]}'
                    )
                
                # Check color mode
                if mode not in ['RGB', 'L', 'RGBA']:
                    validation_result['warnings'].append(f'Unusual color mode: {mode}')
                
                # Check aspect ratio (reasonable bounds)
                aspect_ratio = width / height
                if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                    validation_result['warnings'].append(f'Extreme aspect ratio: {aspect_ratio:.2f}')
            
            # Additional validation with OpenCV (secondary check)
            try:
                cv_image = cv2.imread(str(image_path))
                if cv_image is None:
                    validation_result['errors'].append('OpenCV cannot read image')
                else:
                    cv_height, cv_width = cv_image.shape[:2]
                    if cv_width != width or cv_height != height:
                        validation_result['warnings'].append(
                            f'PIL/OpenCV size mismatch: PIL={width}x{height}, CV={cv_width}x{cv_height}'
                        )
            except Exception as cv_error:
                validation_result['warnings'].append(f'OpenCV validation failed: {str(cv_error)}')
            
            # If no errors, mark as valid
            validation_result['valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['errors'].append(f'Image validation error: {str(e)}')
            logger.warning(f"Validation failed for {image_path}: {e}")
        
        return validation_result
    
    def validate_directory(self, directory: Union[str, Path], 
                          recursive: bool = False) -> Dict[str, Any]:
        """
        Validate all images in a directory.
        
        Performs batch validation of all image files in a directory,
        providing comprehensive statistics and detailed results for
        scientific data quality assessment.
        
        Args:
            directory: Path to directory containing images
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary with directory validation results
        """
        directory = Path(directory)
        
        if not directory.exists():
            return {
                'directory': str(directory),
                'exists': False,
                'total_files': 0,
                'valid_files': 0,
                'invalid_files': 0,
                'validation_rate': 0.0,
                'errors': ['Directory does not exist'],
                'warnings': [],
                'detailed_results': []
            }
        
        if not directory.is_dir():
            return {
                'directory': str(directory),
                'exists': True,
                'total_files': 0,
                'valid_files': 0,
                'invalid_files': 0,
                'validation_rate': 0.0,
                'errors': ['Path is not a directory'],
                'warnings': [],
                'detailed_results': []
            }
        
        # Find all potential image files
        image_files = []
        search_pattern = "**/*" if recursive else "*"
        
        for ext in self.valid_extensions:
            # Case-insensitive search
            image_files.extend(directory.glob(f'{search_pattern}{ext}'))
            image_files.extend(directory.glob(f'{search_pattern}{ext.upper()}'))
        
        # Remove duplicates while preserving order
        image_files = list(dict.fromkeys(image_files))
        
        logger.info(f"Found {len(image_files)} potential image files in {directory}")
        
        valid_files = 0
        invalid_files = 0
        detailed_results = []
        total_size = 0
        
        # Add progress bar for validation
        from tqdm import tqdm
        
        for image_file in tqdm(image_files, desc=f"Validating {directory.name}", unit="img"):
            if image_file.is_file():  # Ensure it's actually a file
                result = self.validate_image_file(image_file)
                detailed_results.append(result)
                
                if result['valid']:
                    valid_files += 1
                    if 'properties' in result and 'file_size' in result['properties']:
                        total_size += result['properties']['file_size']
                else:
                    invalid_files += 1
        
        total_files = len(image_files)
        validation_rate = valid_files / total_files if total_files > 0 else 0.0
        
        return {
            'directory': str(directory),
            'exists': True,
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'validation_rate': validation_rate,
            'total_size_mb': total_size / (1024 * 1024),
            'errors': [],
            'warnings': [] if validation_rate > 0.95 else [f'Low validation rate: {validation_rate:.2%}'],
            'detailed_results': detailed_results
        }
    
    def get_validation_summary(self, validation_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics from validation results.
        
        Args:
            validation_results: List of validation result dictionaries
            
        Returns:
            Summary statistics dictionary
        """
        if not validation_results:
            return {
                'total_images': 0,
                'valid_images': 0,
                'validation_rate': 0.0,
                'common_errors': [],
                'size_statistics': {}
            }
        
        total_images = len(validation_results)
        valid_images = sum(1 for r in validation_results if r.get('valid', False))
        
        # Collect all errors
        all_errors = []
        for result in validation_results:
            all_errors.extend(result.get('errors', []))
        
        # Count error frequency
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # Get most common errors
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Size statistics
        sizes = []
        for result in validation_results:
            if result.get('valid') and 'properties' in result:
                props = result['properties']
                if 'width' in props and 'height' in props:
                    sizes.append((props['width'], props['height']))
        
        size_stats = {}
        if sizes:
            widths, heights = zip(*sizes)
            size_stats = {
                'min_width': min(widths),
                'max_width': max(widths),
                'mean_width': np.mean(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'mean_height': np.mean(heights)
            }
        
        return {
            'total_images': total_images,
            'valid_images': valid_images,
            'validation_rate': valid_images / total_images,
            'common_errors': common_errors,
            'size_statistics': size_stats
        }


class SystemMonitor:
    """
    Monitor system resources during training.
    
    Optimized for Intel Core Ultra 7 monitoring as specified in the hardware
    configuration. Provides real-time system metrics and resource utilization
    tracking for scientific workload optimization.
    """
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get comprehensive system information for Intel Core Ultra 7.
        
        Returns:
            Dictionary with system information including CPU, memory, and disk usage
        """
        try:
            # CPU information
            cpu_info = {
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_percent': psutil.cpu_percent(interval=1, percpu=True),
                'cpu_freq': None,
                'load_average': None
            }
            
            # CPU frequency (if available)
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_info['cpu_freq'] = {
                        'current': cpu_freq.current,
                        'min': cpu_freq.min,
                        'max': cpu_freq.max
                    }
            except Exception:
                pass
            
            # Load average (Unix systems)
            try:
                if hasattr(os, 'getloadavg'):
                    cpu_info['load_average'] = os.getloadavg()
            except Exception:
                pass
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free,
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3)
            }
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100,
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3)
            }
            
            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }
    
    @staticmethod
    def check_system_requirements(config) -> Dict[str, bool]:
        """
        Check if system meets minimum requirements for training.
        
        Validates system capabilities against the Intel Core Ultra 7 configuration
        specified in the project requirements.
        
        Args:
            config: Configuration object with hardware requirements
            
        Returns:
            Dictionary with requirement check results
        """
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count(logical=True)
            disk_free = psutil.disk_usage('/').free
            
            # Get requirements from config
            required_memory_gb = config.get('hardware.memory_gb', 16)
            required_cpu_cores = config.get('hardware.cpu_cores_logical', 8)
            required_disk_gb = 50  # Minimum disk space in GB
            
            checks = {
                'memory_sufficient': memory.total >= required_memory_gb * (1024**3),
                'cpu_cores_sufficient': cpu_count >= required_cpu_cores,
                'disk_space_sufficient': disk_free >= required_disk_gb * (1024**3),
                'memory_available': memory.available >= (required_memory_gb * 0.7) * (1024**3),
                'system_ready': True
            }
            
            # Overall system readiness
            checks['system_ready'] = all([
                checks['memory_sufficient'],
                checks['cpu_cores_sufficient'], 
                checks['disk_space_sufficient'],
                checks['memory_available']
            ])
            
            # Add detailed information
            checks['details'] = {
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'cpu_count': cpu_count,
                'disk_free_gb': disk_free / (1024**3),
                'memory_usage_percent': memory.percent
            }
            
            return checks
            
        except Exception as e:
            logger.error(f"System requirements check failed: {e}")
            return {
                'system_ready': False,
                'error': str(e),
                'memory_sufficient': False,
                'cpu_cores_sufficient': False,
                'disk_space_sufficient': False,
                'memory_available': False
            }
    
    @staticmethod
    def monitor_training_resources(log_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Monitor system resources during training with logging.
        
        Args:
            log_file: Optional file to append monitoring data
            
        Returns:
            Current resource utilization metrics
        """
        try:
            metrics = SystemMonitor.get_system_info()
            
            # Add training-specific metrics
            training_metrics = {
                'cpu_usage_mean': np.mean(metrics['cpu']['cpu_percent']),
                'cpu_usage_max': np.max(metrics['cpu']['cpu_percent']),
                'memory_usage_percent': metrics['memory']['percent'],
                'memory_available_gb': metrics['memory']['available_gb'],
                'disk_usage_percent': metrics['disk']['percent'],
                'timestamp': metrics['timestamp']
            }
            
            # Log to file if specified
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                with AtomicFileWriter.atomic_write(log_path, 'a') as f:
                    f.write(f"{json.dumps(training_metrics)}\n")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
            return {'error': str(e), 'timestamp': pd.Timestamp.now().isoformat()}


class FileHasher:
    """
    Generate and verify file hashes for data integrity tracking.
    
    This class provides cryptographic hash generation for ensuring data integrity
    throughout the scientific workflow. Essential for validating that data has not
    been corrupted during transfer or processing.
    """
    
    @staticmethod
    def generate_file_hash(filepath: Union[str, Path], 
                          algorithm: str = 'sha256') -> str:
        """
        Generate cryptographic hash for a file.
        
        Args:
            filepath: Path to file
            algorithm: Hashing algorithm ('sha256', 'md5', 'sha1')
            
        Returns:
            Hexadecimal hash string
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If algorithm is not supported
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not filepath.is_file():
            raise ValueError(f"Path is not a file: {filepath}")
        
        try:
            hash_obj = hashlib.new(algorithm)
        except ValueError:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        try:
            with open(filepath, 'rb') as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to hash file {filepath}: {e}")
            raise
    
    @staticmethod
    def generate_directory_manifest(directory: Union[str, Path], 
                                  algorithm: str = 'sha256') -> Dict[str, str]:
        """
        Generate hash manifest for all files in directory.
        
        Creates a comprehensive manifest mapping relative file paths to their
        cryptographic hashes. Essential for data integrity verification.
        
        Args:
            directory: Path to directory
            algorithm: Hashing algorithm to use
            
        Returns:
            Dictionary mapping relative file paths to hashes
        """
        directory = Path(directory)
        manifest = {}
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return manifest
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    try:
                        relative_path = file_path.relative_to(directory)
                        file_hash = FileHasher.generate_file_hash(file_path, algorithm)
                        manifest[str(relative_path)] = file_hash
                        
                    except Exception as e:
                        logger.warning(f"Could not hash {file_path}: {e}")
                        manifest[str(relative_path)] = f"ERROR: {str(e)}"
            
            logger.info(f"Generated manifest for {len(manifest)} files in {directory}")
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to generate directory manifest: {e}")
            raise
    
    @staticmethod
    def save_manifest(manifest: Dict[str, str], 
                     output_path: Union[str, Path]) -> None:
        """
        Save manifest to JSON file with atomic write operation.
        
        Args:
            manifest: Dictionary mapping file paths to hashes
            output_path: Path to save manifest file
        """
        try:
            with AtomicFileWriter.atomic_write(output_path) as f:
                json.dump(
                    manifest, 
                    f, 
                    indent=2, 
                    sort_keys=True,
                    ensure_ascii=False
                )
            
            logger.info(f"Saved manifest with {len(manifest)} entries to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save manifest to {output_path}: {e}")
            raise
    
    @staticmethod
    def verify_manifest(manifest_path: Union[str, Path], 
                       base_directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Verify files against a manifest.
        
        Args:
            manifest_path: Path to manifest file
            base_directory: Base directory for relative paths in manifest
            
        Returns:
            Verification results dictionary
        """
        manifest_path = Path(manifest_path)
        base_directory = Path(base_directory)
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load manifest: {e}")
        
        results = {
            'total_files': len(manifest),
            'verified_files': 0,
            'missing_files': 0,
            'corrupted_files': 0,
            'errors': [],
            'missing_list': [],
            'corrupted_list': []
        }
        
        for relative_path, expected_hash in manifest.items():
            if expected_hash.startswith("ERROR:"):
                results['errors'].append(f"{relative_path}: {expected_hash}")
                continue
                
            file_path = base_directory / relative_path
            
            if not file_path.exists():
                results['missing_files'] += 1
                results['missing_list'].append(str(relative_path))
                continue
            
            try:
                actual_hash = FileHasher.generate_file_hash(file_path)
                if actual_hash == expected_hash:
                    results['verified_files'] += 1
                else:
                    results['corrupted_files'] += 1
                    results['corrupted_list'].append(str(relative_path))
                    
            except Exception as e:
                results['errors'].append(f"{relative_path}: {str(e)}")
        
        results['verification_rate'] = results['verified_files'] / results['total_files']
        
        return results
