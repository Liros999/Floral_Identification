"""
Hard negative mining system for Foundational Flower Detector.

This module implements the automated hard negative mining functionality as
specified in the Code_Structure.txt documentation. It uses trained models to
find false positives in background images, creating a queue for human verification.

Key Features:
- Automated false positive detection on background images
- High-confidence threshold filtering (>90% confidence)
- Atomic file operations for thread-safe queue management
- Integration with human verification workflow
- Scientific methodology for systematic bias reduction

References:
- Code_Structure.txt: Detailed find_hard_negatives.py specifications
- Hard negative mining methodology for object detection
- Human-in-the-loop machine learning best practices

Author: Foundational Flower Detector Team
Date: September 2025
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import cv2
from PIL import Image
from tqdm import tqdm

# Project imports
from ..config import Config
from ..models.model_config import ModelFactory
from ..data_preparation.utils import AtomicFileWriter, SystemMonitor
from ..data_preparation.build_dataset import DatasetBuilder

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Automated hard negative mining system.
    
    This class implements the hard negative mining functionality specified in
    the Code_Structure.txt documentation. It loads the best trained model and
    runs inference on background images to identify false positives that will
    be queued for human verification.
    
    The system follows the scientific methodology of systematic bias reduction
    through iterative training with confirmed hard negatives.
    """
    
    def __init__(self, config: Config):
        """
        Initialize hard negative miner.
        
        Args:
            config: Configuration object with mining parameters
        """
        self.config = config
        self.mining_config = config.get('hard_negative_mining', {})
        self.paths = config.get_data_paths()
        
        # Mining parameters
        self.confidence_threshold = self.mining_config.get('confidence_threshold', 0.90)
        self.iou_threshold = self.mining_config.get('iou_threshold', 0.3)
        self.max_fps_per_image = self.mining_config.get('max_false_positives_per_image', 5)
        self.scan_batch_size = self.mining_config.get('background_scan_batch_size', 10)
        self.scan_limit = self.mining_config.get('background_scan_limit', 50000)
        
        # File paths
        self.verification_queue_file = self.paths['base'] / self.mining_config.get(
            'verification_queue_file', 'verification_queue.json'
        )
        self.mining_log_file = self.paths['base'] / self.mining_config.get(
            'mining_log_file', 'hard_negative_mining.log'
        )
        
        # Model state
        self.model = None
        self.model_version = None
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        logger.info(f"Initialized hard negative miner: "
                   f"confidence_threshold={self.confidence_threshold}, "
                   f"scan_limit={self.scan_limit}")
    
    def load_best_model_from_registry(self, model_path: Optional[str] = None) -> tf.keras.Model:
        """
        Load the best trained model for hard negative mining.
        
        This function implements the model loading functionality specified in
        Code_Structure.txt, fetching the best performing model for inference.
        
        Args:
            model_path: Optional explicit path to model weights
            
        Returns:
            Loaded and compiled model
            
        Raises:
            FileNotFoundError: If no suitable model found
        """
        logger.info("Loading best model for hard negative mining")
        
        if model_path and Path(model_path).exists():
            weights_path = Path(model_path)
            logger.info(f"Using explicit model path: {weights_path}")
        else:
            # Find best model in checkpoints directory
            checkpoints_dir = self.paths['checkpoints']
            
            # Look for best model first
            best_model_path = checkpoints_dir / 'best_model.h5'
            final_model_path = checkpoints_dir / 'final_model.h5'
            
            if best_model_path.exists():
                weights_path = best_model_path
                logger.info("Using best_model.h5")
            elif final_model_path.exists():
                weights_path = final_model_path
                logger.info("Using final_model.h5")
            else:
                # Look for any .h5 files
                model_files = list(checkpoints_dir.glob('*.h5'))
                if model_files:
                    weights_path = sorted(model_files)[-1]  # Use most recent
                    logger.info(f"Using most recent model: {weights_path.name}")
                else:
                    raise FileNotFoundError(f"No model weights found in {checkpoints_dir}")
        
        # Create model factory
        model_factory = ModelFactory(self.config)
        
        # Create model for inference
        model = model_factory.create_model(mode='inference')
        
        # Load weights
        try:
            model.load_weights(str(weights_path))
            logger.info(f"Successfully loaded model weights from {weights_path}")
            
            # Extract version info
            self.model_version = self._extract_model_version(weights_path)
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
        
        self.model = model
        return model
    
    def _extract_model_version(self, weights_path: Path) -> str:
        """
        Extract model version information from path and metadata.
        
        Args:
            weights_path: Path to model weights
            
        Returns:
            Version string
        """
        # Try to extract from filename
        if 'best_model' in weights_path.name:
            version = 'best'
        elif 'final_model' in weights_path.name:
            version = 'final'
        else:
            version = weights_path.stem
        
        # Add timestamp
        try:
            stat = weights_path.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime)
            version = f"{version}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        except Exception:
            pass
        
        return version
    
    def scan_background_images(self, background_dirs: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
        """
        Scan background images for false positives.
        
        This function implements the core scanning functionality specified in
        Code_Structure.txt. It iterates through background images, runs inference,
        and identifies false positives based on confidence thresholds.
        
        Args:
            background_dirs: Optional list of directories to scan
            
        Returns:
            List of false positive detections
        """
        logger.info("Starting background image scanning for false positives")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_best_model_from_registry first.")
        
        # Determine background directories to scan
        if background_dirs is None:
            background_dirs = self._get_background_directories()
        
        # Collect background images
        background_images = self._collect_background_images(background_dirs)
        
        if not background_images:
            logger.warning("No background images found to scan")
            return []
        
        # Limit scan if needed
        if len(background_images) > self.scan_limit:
            background_images = background_images[:self.scan_limit]
            logger.info(f"Limited scan to {self.scan_limit} images")
        
        logger.info(f"Scanning {len(background_images)} background images")
        
        # Scan for false positives
        false_positives = []
        processed_count = 0
        start_time = time.time()
        
        # Process in batches with progress bar
        total_batches = (len(background_images) + self.scan_batch_size - 1) // self.scan_batch_size
        logger.info(f"Processing {total_batches} batches of {self.scan_batch_size} images each")
        
        for i in tqdm(range(0, len(background_images), self.scan_batch_size), 
                     desc="Scanning background images", 
                     unit="batch",
                     total=total_batches):
            batch = background_images[i:i + self.scan_batch_size]
            batch_fps = self._process_image_batch(batch)
            false_positives.extend(batch_fps)
            
            processed_count += len(batch)
            
            # Log progress
            if processed_count % 100 == 0 or processed_count == len(background_images):
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {processed_count}/{len(background_images)} images "
                           f"({rate:.1f} img/sec), found {len(false_positives)} false positives")
        
        # Log mining results
        self._log_mining_results(background_images, false_positives)
        
        logger.info(f"Scanning completed: {len(false_positives)} false positives found "
                   f"from {len(background_images)} images")
        
        return false_positives
    
    def _get_background_directories(self) -> List[Path]:
        """Get list of background directories to scan."""
        background_dirs = []
        
        # Primary negative images directory
        negative_dir = self.paths['negative_images']
        if negative_dir.exists():
            background_dirs.append(negative_dir)
        
        # Additional COCO background directories
        coco_dirs = [
            self.paths['raw_data'] / 'coco_images',
            self.paths['raw_data'] / 'background_images',
            self.paths['base'] / 'coco_images'
        ]
        
        for coco_dir in coco_dirs:
            if coco_dir.exists():
                background_dirs.append(coco_dir)
        
        # Filter out directories that don't exist
        existing_dirs = [d for d in background_dirs if d.exists()]
        
        if not existing_dirs:
            logger.warning("No background directories found for scanning")
        
        return existing_dirs
    
    def _collect_background_images(self, background_dirs: List[Path]) -> List[Path]:
        """
        Collect all background images from specified directories.
        
        Args:
            background_dirs: Directories to search for images
            
        Returns:
            List of image paths
        """
        image_extensions = self.config.get('data.image_extensions', ['.jpg', '.jpeg', '.png'])
        background_images = []
        
        for directory in background_dirs:
            logger.info(f"Collecting images from {directory}")
            
            for ext in image_extensions:
                # Case-insensitive search
                background_images.extend(directory.glob(f'*{ext}'))
                background_images.extend(directory.glob(f'*{ext.upper()}'))
        
        # Remove duplicates and sort
        background_images = sorted(list(set(background_images)))
        
        # Filter out invalid images
        valid_images = []
        for img_path in background_images:
            if self._is_valid_image(img_path):
                valid_images.append(img_path)
        
        logger.info(f"Collected {len(valid_images)} valid background images")
        return valid_images
    
    def _is_valid_image(self, image_path: Path) -> bool:
        """
        Check if image is valid for processing.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid
        """
        try:
            # Check file size
            if image_path.stat().st_size < 1000:  # Less than 1KB
                return False
            
            # Try to open with PIL
            with Image.open(image_path) as img:
                # Check dimensions
                width, height = img.size
                min_size = self.config.get('data.min_image_size', [224, 224])
                if width < min_size[0] or height < min_size[1]:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _process_image_batch(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process a batch of images for false positive detection.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            List of false positive detections
        """
        batch_fps = []
        
        for image_path in image_paths:
            try:
                fps = self._process_single_image(image_path)
                batch_fps.extend(fps)
                
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
        
        return batch_fps
    
    def _process_single_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Process single image for false positive detection.
        
        Args:
            image_path: Path to image
            
        Returns:
            List of false positive detections for this image
        """
        # Load and preprocess image
        image = self._load_and_preprocess_image(image_path)
        if image is None:
            return []
        
        # Run inference
        predictions = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        
        # Extract false positives
        false_positives = self._extract_false_positives(image_path, predictions, image.shape)
        
        return false_positives
    
    def _load_and_preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image array or None if failed
        """
        try:
            # Load with OpenCV for consistency
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            target_size = self.config.get('data.target_input_size', [224, 224])
            image = cv2.resize(image, tuple(target_size))
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def _extract_false_positives(self, image_path: Path, predictions: Dict[str, np.ndarray],
                                image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """
        Extract false positive detections from model predictions.
        
        This function implements the false positive extraction logic specified
        in Code_Structure.txt with mathematical threshold checking:
        Flag if max(P(cj)) > T_confidence
        
        Args:
            image_path: Path to source image
            predictions: Model predictions
            image_shape: Original image shape
            
        Returns:
            List of false positive detections
        """
        false_positives = []
        
        try:
            # Extract classification predictions
            if isinstance(predictions, dict):
                classification_probs = predictions.get('classification', predictions.get('output_1'))
            else:
                classification_probs = predictions[0] if len(predictions) > 0 else predictions
            
            if classification_probs is None:
                return false_positives
            
            # Get flower class probability (assuming class 1 is flower)
            if len(classification_probs.shape) == 2:  # Batch dimension
                flower_prob = classification_probs[0, 1] if classification_probs.shape[1] > 1 else classification_probs[0, 0]
            else:
                flower_prob = classification_probs[1] if len(classification_probs) > 1 else classification_probs[0]
            
            # Check if confidence exceeds threshold (this is a false positive)
            if flower_prob > self.confidence_threshold:
                # Create false positive record
                fp_record = {
                    'image_path': str(image_path),
                    'confidence': float(flower_prob),
                    'detection_type': 'classification',
                    'scan_timestamp': datetime.now().isoformat(),
                    'model_version': self.model_version,
                    'threshold_used': self.confidence_threshold
                }
                
                # Add bounding box if available (simplified - whole image)
                height, width = image_shape[:2]
                fp_record['bbox'] = {
                    'x': width * 0.1,
                    'y': height * 0.1,
                    'width': width * 0.8,
                    'height': height * 0.8,
                    'format': 'xywh'
                }
                
                false_positives.append(fp_record)
                
                logger.debug(f"False positive detected: {image_path.name} "
                           f"(confidence: {flower_prob:.3f})")
        
        except Exception as e:
            logger.warning(f"Failed to extract predictions for {image_path}: {e}")
        
        return false_positives
    
    def _log_mining_results(self, scanned_images: List[Path], false_positives: List[Dict[str, Any]]):
        """
        Log mining results for tracking and analysis.
        
        Args:
            scanned_images: List of images that were scanned
            false_positives: List of false positive detections
        """
        mining_result = {
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version,
            'scan_parameters': {
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold,
                'max_fps_per_image': self.max_fps_per_image,
                'scan_limit': self.scan_limit
            },
            'scan_results': {
                'total_images_scanned': len(scanned_images),
                'false_positives_found': len(false_positives),
                'false_positive_rate': len(false_positives) / len(scanned_images) if scanned_images else 0,
                'images_with_fps': len(set(fp['image_path'] for fp in false_positives))
            },
            'system_info': self.system_monitor.get_system_info()
        }
        
        # Log to mining log file
        with AtomicFileWriter.atomic_write(self.mining_log_file, 'a') as f:
            f.write(json.dumps(mining_result) + '\n')
        
        logger.info(f"Mining results logged: {len(false_positives)} FPs from {len(scanned_images)} images")
    
    def create_verification_queue(self, false_positives: List[Dict[str, Any]]) -> Path:
        """
        Create verification queue for human review.
        
        This function implements the verification queue creation specified in
        Code_Structure.txt, preparing false positives for human verification
        in the Streamlit UI.
        
        Args:
            false_positives: List of false positive detections
            
        Returns:
            Path to created verification queue file
        """
        logger.info(f"Creating verification queue with {len(false_positives)} items")
        
        # Prepare queue data
        queue_data = {
            'created_at': datetime.now().isoformat(),
            'model_version': self.model_version,
            'total_items': len(false_positives),
            'confidence_threshold': self.confidence_threshold,
            'status': 'pending_verification',
            'false_positives': false_positives,
            'verification_progress': {
                'confirmed': 0,
                'rejected': 0,
                'pending': len(false_positives)
            }
        }
        
        # Add metadata for UI
        queue_data['ui_metadata'] = {
            'images_per_page': self.config.get('ui.images_per_page', 10),
            'display_size': self.config.get('ui.max_image_display_size', [800, 600]),
            'instructions': "Review each detection and confirm if it's truly a false positive (not a flower)."
        }
        
        # Save queue atomically
        with AtomicFileWriter.atomic_write(self.verification_queue_file) as f:
            json.dump(queue_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Verification queue created: {self.verification_queue_file}")
        return self.verification_queue_file
    
    def mine_hard_negatives(self, model_path: Optional[str] = None,
                           background_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main orchestration function for hard negative mining.
        
        This function implements the main() execution block specified in
        Code_Structure.txt, orchestrating model loading, background scanning,
        and verification queue creation.
        
        Args:
            model_path: Optional path to specific model
            background_dirs: Optional list of background directories
            
        Returns:
            Mining results and queue information
        """
        logger.info("Starting hard negative mining process")
        start_time = time.time()
        
        try:
            # Step 1: Load model
            logger.info("Step 1/4: Loading best model from registry")
            self.load_best_model_from_registry(model_path)
            
            # Step 2: Prepare background directories
            logger.info("Step 2/4: Preparing background directories")
            bg_dirs = None
            if background_dirs:
                bg_dirs = [Path(d) for d in background_dirs]
            
            # Step 3: Scan for false positives
            logger.info("Step 3/4: Scanning background images for false positives")
            false_positives = self.scan_background_images(bg_dirs)
            
            # Check convergence criteria
            convergence_check = self._check_convergence(false_positives)
            
            # Step 4: Create verification queue
            logger.info("Step 4/4: Creating verification queue")
            queue_file = None
            if false_positives:
                queue_file = self.create_verification_queue(false_positives)
            
            # Calculate mining time
            mining_time = time.time() - start_time
            
            # Compile results
            results = {
                'mining_time_seconds': mining_time,
                'model_version': self.model_version,
                'false_positives_found': len(false_positives),
                'convergence_check': convergence_check,
                'verification_queue_file': str(queue_file) if queue_file else None,
                'scan_parameters': {
                    'confidence_threshold': self.confidence_threshold,
                    'scan_limit': self.scan_limit,
                    'batch_size': self.scan_batch_size
                },
                'next_steps': self._get_next_steps(false_positives, convergence_check)
            }
            
            logger.info("Hard negative mining completed successfully!")
            logger.info(f"Found {len(false_positives)} false positives in {mining_time:.1f} seconds")
            
            if convergence_check['converged']:
                logger.info("🎉 Convergence achieved! Hard negative mining cycle complete.")
            else:
                logger.info(f"Continue mining: {convergence_check['reason']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Hard negative mining failed: {e}")
            raise
    
    def _check_convergence(self, false_positives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if hard negative mining has converged.
        
        Args:
            false_positives: Current false positives found
            
        Returns:
            Convergence check results
        """
        exit_threshold = self.mining_config.get('cycle_exit_threshold', 50)
        min_fps_per_cycle = self.mining_config.get('min_hard_negatives_per_cycle', 50)
        
        convergence_check = {
            'converged': False,
            'reason': '',
            'fps_found': len(false_positives),
            'exit_threshold': exit_threshold,
            'min_threshold': min_fps_per_cycle
        }
        
        if len(false_positives) < exit_threshold:
            convergence_check['converged'] = True
            convergence_check['reason'] = f"Found {len(false_positives)} FPs < threshold {exit_threshold}"
        elif len(false_positives) < min_fps_per_cycle:
            convergence_check['reason'] = f"Low FP count: {len(false_positives)} < {min_fps_per_cycle}"
        else:
            convergence_check['reason'] = f"Continue mining: {len(false_positives)} FPs found"
        
        return convergence_check
    
    def _get_next_steps(self, false_positives: List[Dict[str, Any]], 
                       convergence_check: Dict[str, Any]) -> List[str]:
        """Get recommended next steps based on mining results."""
        next_steps = []
        
        if false_positives:
            next_steps.append("1. Launch verification UI to review false positives")
            next_steps.append("2. Confirm true false positives through human verification")
        
        if convergence_check['converged']:
            next_steps.extend([
                "3. 🎉 Mining converged - evaluate final model performance",
                "4. Consider deploying model or moving to next research phase"
            ])
        else:
            next_steps.extend([
                "3. After verification, rebuild dataset with confirmed hard negatives",
                "4. Retrain model with augmented negative examples",
                "5. Repeat mining cycle until convergence"
            ])
        
        return next_steps


def main():
    """
    Main entry point for hard negative mining script.
    
    This function can be called directly or used as a command-line script
    for mining hard negatives independently.
    """
    import sys
    import argparse
    
    # Setup argument parsing
    parser = argparse.ArgumentParser(description='Mine Hard Negatives for Foundational Flower Detector')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model-path', type=str, help='Path to model weights')
    parser.add_argument('--background-dirs', nargs='+', help='Background directories to scan')
    parser.add_argument('--confidence-threshold', type=float, help='Confidence threshold for FPs')
    parser.add_argument('--scan-limit', type=int, help='Maximum images to scan')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load configuration
        from ..config import Config
        config = Config(args.config)
        
        # Override config with command line arguments
        if args.confidence_threshold:
            config.set('hard_negative_mining.confidence_threshold', args.confidence_threshold)
        if args.scan_limit:
            config.set('hard_negative_mining.background_scan_limit', args.scan_limit)
        
        # Create miner
        miner = HardNegativeMiner(config)
        
        # Run mining
        results = miner.mine_hard_negatives(
            model_path=args.model_path,
            background_dirs=args.background_dirs
        )
        
        # Print results
        logger.info("Mining Results:")
        logger.info(f"False positives found: {results['false_positives_found']}")
        logger.info(f"Mining time: {results['mining_time_seconds']:.1f} seconds")
        logger.info(f"Converged: {results['convergence_check']['converged']}")
        
        if results['verification_queue_file']:
            logger.info(f"Verification queue: {results['verification_queue_file']}")
        
        logger.info("Next steps:")
        for step in results['next_steps']:
            logger.info(f"  {step}")
        
    except Exception as e:
        logger.error(f"Hard negative mining failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
