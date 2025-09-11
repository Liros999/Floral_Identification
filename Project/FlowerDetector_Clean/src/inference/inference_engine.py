"""
Production Inference Engine for Flower Detection
Handles single image and batch inference with confidence scores.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FlowerInferenceEngine:
    """
    Production-ready inference engine for flower detection.
    Supports single image and batch processing with comprehensive logging.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', config: Dict[str, Any] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
            config: Optional configuration dictionary
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.config = config or {}
        
        # Load model
        self.model = self._load_model()
        self.transform = self._create_inference_transform()
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        logger.info(f"FlowerInferenceEngine initialized on {self.device}")
        logger.info(f"Model loaded from: {self.model_path}")
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint with weights_only=False for compatibility
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Import model class
        from src.model.flower_classifier import FlowerClassifier
        
        # Create model instance
        model_config = checkpoint.get('config', {}).get('model', {})
        model = FlowerClassifier(
            num_classes=model_config.get('num_classes', 2),
            pretrained=False,  # Don't load pretrained weights for inference
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def _create_inference_transform(self) -> transforms.Compose:
        """Create image preprocessing transform for inference."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Preprocess image for inference."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path string, Path object, or PIL Image")
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict_single(self, image: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Predict flower presence in a single image.
        
        Args:
            image: Image path, Path object, or PIL Image
            
        Returns:
            Dictionary with prediction results and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence = probabilities[0, 1].item()  # Flower class probability
                prediction = 1 if confidence > 0.5 else 0
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            # Prepare result
            result = {
                'prediction': 'flower' if prediction == 1 else 'background',
                'confidence': confidence,
                'probabilities': {
                    'background': float(probabilities[0, 0]),
                    'flower': float(probabilities[0, 1])
                },
                'inference_time_ms': inference_time * 1000,
                'timestamp': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'device': str(self.device)
            }
            
            logger.debug(f"Single image prediction completed in {inference_time*1000:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error in single image prediction: {e}")
            return {
                'error': str(e),
                'prediction': None,
                'confidence': 0.0,
                'inference_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[Dict[str, Any]]:
        """
        Predict flower presence in a batch of images.
        
        Args:
            images: List of image paths, Path objects, or PIL Images
            
        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()
        results = []
        
        try:
            # Preprocess all images
            input_tensors = []
            for image in images:
                tensor = self._preprocess_image(image)
                input_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(input_tensors, dim=0).to(self.device)
            
            # Run batch inference
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidences = probabilities[:, 1]  # Flower class probabilities
                predictions = (confidences > 0.5).int()
            
            # Prepare results
            for i, (pred, conf, probs) in enumerate(zip(predictions, confidences, probabilities)):
                result = {
                    'prediction': 'flower' if pred.item() == 1 else 'background',
                    'confidence': conf.item(),
                    'probabilities': {
                        'background': float(probs[0]),
                        'flower': float(probs[1])
                    },
                    'batch_index': i,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
            
            # Update performance stats
            batch_time = time.time() - start_time
            self.inference_count += len(images)
            self.total_inference_time += batch_time
            
            logger.info(f"Batch prediction completed: {len(images)} images in {batch_time*1000:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            # Return error results for all images
            return [{
                'error': str(e),
                'prediction': None,
                'confidence': 0.0,
                'batch_index': i,
                'timestamp': datetime.now().isoformat()
            } for i in range(len(images))]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
        
        return {
            'total_inferences': self.inference_count,
            'total_time_seconds': self.total_inference_time,
            'average_time_ms': avg_time * 1000,
            'inferences_per_second': 1.0 / avg_time if avg_time > 0 else 0,
            'device': str(self.device),
            'model_path': str(self.model_path)
        }
    
    def save_predictions(self, predictions: List[Dict[str, Any]], output_path: str) -> None:
        """Save predictions to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Predictions saved to {output_file}")
    
    def benchmark(self, test_images: List[Union[str, Path, Image.Image]], 
                  iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            test_images: List of test images
            iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        logger.info(f"Starting benchmark with {len(test_images)} images, {iterations} iterations")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            self.predict_batch(test_images)
            times.append(time.time() - start_time)
        
        times = sorted(times)
        median_time = times[len(times) // 2]
        avg_time = sum(times) / len(times)
        
        results = {
            'iterations': iterations,
            'images_per_iteration': len(test_images),
            'median_time_ms': median_time * 1000,
            'average_time_ms': avg_time * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'images_per_second': len(test_images) / avg_time,
            'device': str(self.device)
        }
        
        logger.info(f"Benchmark completed: {results['images_per_second']:.2f} images/second")
        return results
