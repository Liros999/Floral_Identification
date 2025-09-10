"""
Centralized configuration management for the Foundational Flower Detector.

This module implements the configuration system as specified in the architecture documents,
optimized for CPU training with Intel Core Ultra 7 and Google Drive data integration.

References:
- Project architecture documents: General Architecture, Model Structure and Pipeline
- CPU optimization strategies for scientific computing
- Reproducibility standards for scientific software

Author: Foundational Flower Detector Team
Date: September 2025
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class Config:
    """
    Centralized configuration management for the Foundational Flower Detector.
    
    This class manages all configuration parameters following the scientific rigor
    principles outlined in the project documentation. It provides:
    
    - CPU-optimized training parameters for Intel Core Ultra 7
    - Google Drive data path integration
    - Reproducibility settings with deterministic behavior
    - Hardware-aware optimization parameters
    - Scientific evaluation thresholds
    
    The configuration follows the architecture decisions documented in:
    - General Architecture(Gemini 8_9_25).txt
    - Model_Structure_and_Pipeline.txt
    - Code_Structure.txt
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to custom configuration file.
                        If None, uses default config.yaml in project root.
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = config_path or self.project_root / "config.yaml"
        self.config = self._load_config()
        
        # Set up logging after config is loaded
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with scientifically validated defaults.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        default_config = self._get_default_config()
        
        # Load from file if exists, otherwise use defaults
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                
                # Deep merge configurations (file overrides defaults)
                merged_config = self._deep_merge(default_config, file_config)
                logger.info(f"Configuration loaded from {self.config_path}")
                return merged_config
                
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Using default configuration")
                
        else:
            # Save default config for reference
            self._save_config(default_config)
            logger.info(f"Created default configuration at {self.config_path}")
        
        return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Generate default configuration optimized for the project requirements.
        
        This configuration implements the architectural decisions from the
        project documentation, including CPU optimization for Intel Core Ultra 7
        and integration with Google Drive data paths.
        
        Returns:
            Dictionary with default configuration parameters
        """
        return {
            # Project metadata
            'project': {
                'name': 'foundational_flower_detector',
                'version': '1.0.0',
                'description': 'High-precision flower detection with hard negative mining',
                'phase': 'foundational',
                'author': 'Foundational Flower Detector Team',
                'created_date': '2025-09-09'
            },
            
            # Data configuration - Google Drive integration
            'data': {
                # Google Drive paths (primary data source)
                'google_drive_base_path': r'G:\My Drive\Floral_Detector\Phase1_Foundational-Detector\Phase1_Data',
                'raw_data_path': r'G:\My Drive\Floral_Detector\Phase1_Foundational-Detector\Phase1_Data\raw_data',
                'processed_data_path': r'G:\My Drive\Floral_Detector\Phase1_Foundational-Detector\Phase1_Data\processed_data',
                'reports_path': r'G:\My Drive\Floral_Detector\Phase1_Foundational-Detector\Phase1_Data\reports',
                
                # Local paths (for working copies and outputs)
                'local_data_path': str(self.project_root / 'data'),
                'local_raw_path': str(self.project_root / 'data' / 'raw'),
                'local_processed_path': str(self.project_root / 'data' / 'processed'),
                
                # Dataset specifications
                'positive_images_subpath': 'positive_images',
                'negative_images_subpath': 'negative_images',
                'annotations_subpath': 'annotations',
                'metadata_subpath': 'metadata',
                
                # Dataset splits (deterministic)
                'train_split': 0.7,
                'val_split': 0.2,
                'test_split': 0.1,
                
                # Reproducibility (Decision A3)
                'global_random_seed': 42,
                
                # Image specifications
                'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
                'min_image_size': [224, 224],
                'max_image_size': [1024, 1024],
                'target_input_size': [224, 224],  # CPU-optimized smaller input
                
                # Dataset size management
                'max_dataset_size': {
                    'positive': 1000,
                    'negative': 2000,
                    'validation_positive': 200,
                    'validation_negative': 400
                }
            },
            
            # CPU-optimized training configuration for Intel Core Ultra 7
            'training': {
                # Batch configuration (optimized for 32GB RAM)
                'batch_size': 2,                    # Small batch for CPU training
                'validation_batch_size': 1,
                'gradient_accumulation_steps': 8,   # Simulate larger batch
                
                # Learning parameters (conservative for stability)
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'momentum': 0.9,
                'gradient_clipping': 1.0,
                
                # Training duration
                'epochs': 50,                       # More epochs with smaller batches
                'patience': 10,                     # Early stopping patience
                'min_epochs': 10,                   # Minimum training duration
                
                # CPU optimization (Intel Core Ultra 7: 16 logical cores)
                'num_workers': 8,                   # Half logical cores for data loading
                'prefetch_factor': 2,
                'pin_memory': False,                # No GPU benefit
                'persistent_workers': True,
                'drop_last': True,                  # Ensure consistent batch sizes
                
                # Optimizer configuration
                'optimizer': 'adam',
                'scheduler': 'plateau',
                'scheduler_patience': 5,
                'scheduler_factor': 0.5,
                'min_lr': 1e-6,
                
                # Mixed precision and compilation (CPU settings)
                'mixed_precision': False,           # Not beneficial for CPU
                'compile_model': False,             # Can be slower on CPU
                
                # Checkpointing and logging
                'checkpoint_frequency': 5,
                'save_best_only': True,
                'save_last': True,
                'monitor_metric': 'val_precision',
                'monitor_mode': 'max',
                
                # Validation frequency
                'validation_frequency': 1,          # Every epoch
                'log_frequency': 100               # Every N steps
            },
            
            # Mask R-CNN model configuration (following He et al. 2017)
            'model': {
                'architecture': 'mask_rcnn',
                'backbone': 'resnet50',             # Smaller backbone for CPU efficiency
                'pretrained': True,
                'pretrained_weights': 'coco',       # COCO pre-trained weights
                
                # Class configuration
                'num_classes': 2,                   # background + flower
                'class_names': ['background', 'flower'],
                'class_weights': [1.0, 2.0],      # Higher weight for flower class
                
                # Input configuration (CPU-optimized)
                'input_size': [224, 224],          # Smaller input for CPU training
                'input_channels': 3,
                'input_pixel_mean': [123.675, 116.28, 103.53],   # ImageNet normalization
                'input_pixel_std': [58.395, 57.12, 57.375],
                
                # RPN (Region Proposal Network) configuration
                'rpn_anchor_scales': [32, 64, 128, 256, 512],
                'rpn_anchor_ratios': [0.5, 1.0, 2.0],
                'rpn_train_pre_nms_topN': 2000,
                'rpn_train_post_nms_topN': 1000,
                'rpn_test_pre_nms_topN': 1000,
                'rpn_test_post_nms_topN': 1000,
                'rpn_nms_threshold': 0.7,
                
                # ROI configuration
                'roi_pool_size': 7,
                'mask_pool_size': 14,
                'roi_positive_ratio': 0.25,
                'roi_batch_size': 512,
                'roi_positive_overlap': 0.5,
                'roi_negative_overlap_high': 0.5,
                'roi_negative_overlap_low': 0.1,
                
                # FPN (Feature Pyramid Network) configuration
                'fpn_num_filters': 256,
                'fpn_num_layers': 4
            },
            
            # Hard negative mining configuration (core innovation)
            'hard_negative_mining': {
                # Detection thresholds (high precision focus)
                'confidence_threshold': 0.90,      # High threshold for hard negatives
                'iou_threshold': 0.3,
                'nms_threshold': 0.5,
                
                # Mining parameters
                'max_false_positives_per_image': 5,
                'background_scan_batch_size': 10,
                'min_hard_negatives_per_cycle': 50,
                'max_cycles': 10,
                'cycle_exit_threshold': 50,        # <50 new FPs = convergence
                
                # File management (atomic operations - Decision A1)
                'verification_queue_file': 'verification_queue.json',
                'confirmed_negatives_log': 'confirmed_hard_negatives.log',
                'mining_log_file': 'hard_negative_mining.log',
                
                # Background image scanning
                'background_scan_limit': 50000,    # Max images per scan cycle
                'parallel_scanning': True,
                'scan_batch_workers': 4
            },
            
            # Evaluation configuration (scientific rigor)
            'evaluation': {
                # Success thresholds (from architecture documents)
                'precision_threshold': 0.98,       # Primary success metric
                'recall_threshold': 0.85,          # Secondary success metric
                'f1_threshold': 0.91,             # Derived threshold
                
                # Evaluation parameters
                'iou_threshold': 0.5,              # COCO standard
                'confidence_threshold': 0.5,
                'max_detections': 100,
                
                # Challenge set (Decision A2)
                'challenge_set_size': 100,
                'challenge_set_file': 'challenge_set.json',
                'challenge_evaluation_frequency': 5,  # Every 5 epochs
                
                # Metrics logging
                'detailed_metrics': True,
                'save_predictions': True,
                'save_confusion_matrix': True
            },
            
            # Hardware configuration (Intel Core Ultra 7 optimization)
            'hardware': {
                # CPU specifications
                'cpu_cores_logical': 16,           # Intel Core Ultra 7
                'cpu_cores_physical': 8,
                'memory_gb': 32,
                'cpu_architecture': 'intel_ultra_7',
                
                # Threading configuration
                'intraop_threads': 8,              # TensorFlow intra-op parallelism
                'interop_threads': 4,              # TensorFlow inter-op parallelism
                'mkl_threads': 8,                  # Intel MKL threads
                
                # Optimizations
                'use_multiprocessing': True,
                'enable_mkl': True,                # Intel MKL optimization
                'mkl_dnn': True,                   # Intel oneDNN
                'memory_growth': True,             # TensorFlow memory management
                
                # System monitoring
                'monitor_system': True,
                'memory_limit_gb': 28,             # Leave 4GB for system
                'cpu_usage_threshold': 0.95
            },
            
            # Logging configuration
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'date_format': '%Y-%m-%d %H:%M:%S',
                
                # File logging
                'log_file': 'logs/training.log',
                'max_file_size': '100MB',
                'backup_count': 5,
                'encoding': 'utf-8',
                
                # Console logging
                'console_level': 'INFO',
                'console_format': '%(levelname)s - %(message)s',
                
                # TensorBoard
                'tensorboard_dir': 'logs/tensorboard',
                'tensorboard_update_freq': 100,
                'tensorboard_write_graph': True,
                'tensorboard_write_images': True,
                
                # MLflow (optional for experiment tracking)
                'mlflow_enabled': False,
                'mlflow_tracking_uri': '',
                'mlflow_experiment_name': 'foundational_flower_detector'
            },
            
            # Paths configuration
            'paths': {
                # Project directories
                'project_root': str(self.project_root),
                'source_dir': str(self.project_root / 'src'),
                'tests_dir': str(self.project_root / 'tests'),
                
                # Data directories (local)
                'data_dir': str(self.project_root / 'data'),
                'raw_data_dir': str(self.project_root / 'data' / 'raw'),
                'processed_data_dir': str(self.project_root / 'data' / 'processed'),
                'external_data_dir': str(self.project_root / 'data' / 'external'),
                
                # Output directories
                'models_dir': str(self.project_root / 'models'),
                'checkpoints_dir': str(self.project_root / 'models' / 'checkpoints'),
                'logs_dir': str(self.project_root / 'logs'),
                'reports_dir': str(self.project_root / 'reports'),
                'figures_dir': str(self.project_root / 'reports' / 'figures'),
                
                # Temporary directories
                'temp_dir': str(self.project_root / 'tmp'),
                'cache_dir': str(self.project_root / 'cache')
            },
            
            # Streamlit UI configuration
            'ui': {
                # Server settings
                'port': 8501,
                'host': 'localhost',
                'debug': False,
                
                # UI settings
                'title': 'Foundational Flower Detector - Verification UI',
                'page_icon': '🌸',
                'layout': 'wide',
                'initial_sidebar_state': 'expanded',
                
                # Display settings
                'images_per_page': 10,
                'max_image_display_size': [800, 600],
                'thumbnail_size': [200, 200],
                
                # Interaction settings
                'queue_refresh_interval': 30,      # seconds
                'auto_advance': True,
                'keyboard_shortcuts': True,
                
                # Progress tracking
                'show_progress': True,
                'show_statistics': True,
                'save_session_log': True
            },
            
            # Development and debugging configuration
            'development': {
                'debug': False,
                'verbose': True,
                'profile': False,
                'reproducible': True,
                
                # Testing
                'test_data_size': 100,
                'mock_training': False,
                'fast_dev_run': False,
                
                # Development tools
                'auto_reload': True,
                'jupyter_support': True,
                'notebook_dir': str(self.project_root / 'notebooks'),
                
                # Code quality
                'enable_linting': True,
                'enable_type_checking': True,
                'strict_mode': True
            }
        }
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with dict2 values taking precedence.
        
        Args:
            dict1: Base dictionary
            dict2: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to YAML file with proper formatting.
        
        Args:
            config: Configuration dictionary to save
        """
        try:
            # Ensure directory exists
            os.makedirs(self.config_path.parent, exist_ok=True)
            
            # Save with proper YAML formatting
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config, 
                    f, 
                    default_flow_style=False, 
                    indent=2,
                    sort_keys=False,
                    allow_unicode=True
                )
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration based on config parameters."""
        log_level = getattr(logging, self.get('logging.level', 'INFO').upper())
        log_format = self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=self.get('logging.date_format', '%Y-%m-%d %H:%M:%S')
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config = Config()
            >>> batch_size = config.get('training.batch_size')
            >>> learning_rate = config.get('training.learning_rate', 0.001)
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config_ref = self.config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self) -> None:
        """Save current configuration to file."""
        self._save_config(self.config)
        logger.info(f"Configuration saved to {self.config_path}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration parameters for consistency and correctness.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate data paths
            google_drive_path = Path(self.get('data.google_drive_base_path', ''))
            if not google_drive_path.exists():
                validation_results['errors'].append(f"Google Drive path does not exist: {google_drive_path}")
                validation_results['valid'] = False
            
            # Validate splits sum to 1.0
            train_split = self.get('data.train_split', 0.7)
            val_split = self.get('data.val_split', 0.2)
            test_split = self.get('data.test_split', 0.1)
            
            if abs(train_split + val_split + test_split - 1.0) > 1e-6:
                validation_results['errors'].append("Data splits do not sum to 1.0")
                validation_results['valid'] = False
            
            # Validate hardware configuration
            batch_size = self.get('training.batch_size', 2)
            num_workers = self.get('training.num_workers', 8)
            cpu_cores = self.get('hardware.cpu_cores_logical', 16)
            
            if num_workers > cpu_cores:
                validation_results['warnings'].append(f"num_workers ({num_workers}) > cpu_cores ({cpu_cores})")
            
            if batch_size < 1:
                validation_results['errors'].append("Batch size must be >= 1")
                validation_results['valid'] = False
            
            # Validate threshold values
            precision_threshold = self.get('evaluation.precision_threshold', 0.98)
            if not 0.0 <= precision_threshold <= 1.0:
                validation_results['errors'].append("Precision threshold must be between 0.0 and 1.0")
                validation_results['valid'] = False
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['valid'] = False
        
        return validation_results
    
    def get_data_paths(self) -> Dict[str, Path]:
        """
        Get all relevant data paths as Path objects.
        
        Returns:
            Dictionary mapping path names to Path objects
        """
        base_path = Path(self.get('data.google_drive_base_path'))
        
        return {
            'base': base_path,
            'raw_data': base_path / 'raw_data',
            'positive_images': base_path / 'raw_data' / self.get('data.positive_images_subpath'),
            'negative_images': base_path / 'raw_data' / self.get('data.negative_images_subpath'),
            'processed_data': base_path / 'processed_data',
            'annotations': base_path / 'processed_data' / self.get('data.annotations_subpath'),
            'metadata': base_path / 'raw_data' / self.get('data.metadata_subpath'),
            'reports': base_path / 'reports',
            'local_data': Path(self.get('paths.data_dir')),
            'local_processed': Path(self.get('paths.processed_data_dir')),
            'checkpoints': Path(self.get('paths.checkpoints_dir')),
            'logs': Path(self.get('paths.logs_dir'))
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(project={self.get('project.name')}, version={self.get('project.version')})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Config(config_path='{self.config_path}', loaded={self.config_path.exists()})"
