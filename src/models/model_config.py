"""
Model architecture configuration for Foundational Flower Detector.

This module implements the Mask R-CNN configuration following He et al. (2017)
with optimizations for CPU training on Intel Core Ultra 7. The implementation
focuses on scientific rigor with reproducible model architectures.

Key Features:
- Mask R-CNN architecture based on He et al. (2017)
- CPU-optimized configurations for Intel Core Ultra 7
- ResNet-50 backbone for efficiency vs. accuracy trade-off
- Binary classification: background + flower
- Scientific evaluation metrics integration

References:
- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. ICCV.
- TensorFlow Object Detection API documentation
- CPU optimization strategies for deep learning

Author: Foundational Flower Detector Team
Date: September 2025
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class MaskRCNNConfig:
    """
    Configuration class for Mask R-CNN architecture.
    
    This class implements the Mask R-CNN configuration as specified in the
    architecture documents, with CPU optimizations for Intel Core Ultra 7
    and parameters tuned for high-precision flower detection.
    
    The configuration follows the methodology from He et al. (2017) with
    adaptations for binary classification and CPU training efficiency.
    """
    
    def __init__(self, config):
        """
        Initialize Mask R-CNN configuration.
        
        Args:
            config: Main configuration object with model parameters
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.hardware_config = config.get('hardware', {})
        
        # Core architecture parameters
        self.backbone = self.model_config.get('backbone', 'resnet50')
        self.num_classes = self.model_config.get('num_classes', 2)  # background + flower
        self.input_size = self.model_config.get('input_size', [224, 224])
        self.pretrained = self.model_config.get('pretrained', True)
        
        # CPU optimization settings
        self.cpu_optimized = True
        self.mixed_precision = False  # Not beneficial for CPU
        
        # Feature layer names for FPN
        self.feature_layers = [
            'conv2_block3_out',  # C2 (P2)
            'conv3_block4_out',  # C3 (P3)
            'conv4_block6_out',  # C4 (P4)
            'conv5_block3_out'   # C5 (P5)
        ]
        
        logger.info(f"Initialized Mask R-CNN config: backbone={self.backbone}, "
                   f"num_classes={self.num_classes}, input_size={self.input_size}")
    
    def get_backbone_config(self) -> Dict[str, Any]:
        """
        Get backbone network configuration.
        
        Returns:
            Dictionary with backbone configuration parameters
        """
        backbone_configs = {
            'resnet50': {
                'architecture': 'ResNet50',
                'weights': 'imagenet' if self.pretrained else None,
                'include_top': False,
                'input_shape': (*self.input_size, 3),
                'pooling': None,
                'trainable': True,
                # Keras ResNet50 layer names
                'layers_to_freeze': [
                    'conv1_conv', 'conv1_bn',
                    'conv2_block1_out', 'conv2_block2_out', 'conv2_block3_out'
                ],  # Freeze early layers
                'feature_layers': [
                    'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'
                ]  # FPN feature extraction
            },
            'resnet101': {
                'architecture': 'ResNet101',
                'weights': 'imagenet' if self.pretrained else None,
                'include_top': False,
                'input_shape': (*self.input_size, 3),
                'pooling': None,
                'trainable': True,
                # Keras ResNet101 layer names (follow same stage endpoints)
                'layers_to_freeze': [
                    'conv1_conv', 'conv1_bn',
                    'conv2_block1_out', 'conv2_block2_out', 'conv2_block3_out'
                ],
                'feature_layers': [
                    'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'
                ]
            }
        }
        
        if self.backbone not in backbone_configs:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        config = backbone_configs[self.backbone].copy()
        
        # CPU optimizations
        if self.cpu_optimized:
            # Reduce complexity for CPU training
            config['cpu_optimized'] = True
            
        return config
    
    def get_rpn_config(self) -> Dict[str, Any]:
        """
        Get Region Proposal Network (RPN) configuration.
        
        Returns:
            Dictionary with RPN configuration parameters
        """
        return {
            # Anchor configuration
            'anchor_scales': self.model_config.get('rpn_anchor_scales', [32, 64, 128, 256, 512]),
            'anchor_ratios': self.model_config.get('rpn_anchor_ratios', [0.5, 1.0, 2.0]),
            'anchor_stride': 16,  # Based on backbone feature map stride
            
            # Training parameters
            'train_pre_nms_topN': self.model_config.get('rpn_train_pre_nms_topN', 2000),
            'train_post_nms_topN': self.model_config.get('rpn_train_post_nms_topN', 1000),
            'test_pre_nms_topN': self.model_config.get('rpn_test_pre_nms_topN', 1000),
            'test_post_nms_topN': self.model_config.get('rpn_test_post_nms_topN', 1000),
            
            # NMS parameters
            'nms_threshold': self.model_config.get('rpn_nms_threshold', 0.7),
            'positive_overlap': 0.7,  # IoU threshold for positive anchors
            'negative_overlap': 0.3,  # IoU threshold for negative anchors
            
            # Loss weights
            'loss_weight_cls': 1.0,
            'loss_weight_bbox': 1.0,
            
            # CPU optimizations
            'batch_size': 256 if self.cpu_optimized else 512,
            'positive_fraction': 0.5
        }
    
    def get_roi_config(self) -> Dict[str, Any]:
        """
        Get ROI (Region of Interest) configuration.
        
        Returns:
            Dictionary with ROI configuration parameters
        """
        return {
            # ROI pooling
            'pool_size': self.model_config.get('roi_pool_size', 7),
            'mask_pool_size': self.model_config.get('mask_pool_size', 14),
            
            # ROI sampling
            'batch_size': self.model_config.get('roi_batch_size', 512),
            'positive_ratio': self.model_config.get('roi_positive_ratio', 0.25),
            'positive_overlap': self.model_config.get('roi_positive_overlap', 0.5),
            'negative_overlap_high': self.model_config.get('roi_negative_overlap_high', 0.5),
            'negative_overlap_low': self.model_config.get('roi_negative_overlap_low', 0.1),
            
            # Classification head
            'fc_layers': [1024, 1024] if not self.cpu_optimized else [512, 512],  # Smaller for CPU
            'dropout_rate': 0.5,
            
            # Mask head
            'mask_conv_layers': 4,
            'mask_conv_dim': 256,
            'mask_resolution': self.model_config.get('mask_pool_size', 14),
            
            # Loss weights
            'loss_weight_cls': 1.0,
            'loss_weight_bbox': 1.0,
            'loss_weight_mask': 1.0
        }
    
    def get_fpn_config(self) -> Dict[str, Any]:
        """
        Get Feature Pyramid Network (FPN) configuration.
        
        Returns:
            Dictionary with FPN configuration parameters
        """
        return {
            'num_filters': self.model_config.get('fpn_num_filters', 256),
            'num_layers': self.model_config.get('fpn_num_layers', 4),
            'min_level': 2,  # P2
            'max_level': 5,  # P5
            'add_coarse_level': True,  # P6 for larger objects
            'use_separable_conv': self.cpu_optimized,  # More efficient for CPU
            'activation': 'relu',
            'use_batch_norm': True,
            'batch_norm_momentum': 0.99,
            'batch_norm_epsilon': 1e-3
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training-specific configuration.
        
        Returns:
            Dictionary with training configuration parameters
        """
        return {
            # Learning rate schedule
            'learning_rate': self.training_config.get('learning_rate', 0.001),
            'learning_rate_schedule': 'cosine_decay',
            'warmup_steps': 1000,
            'total_steps': self.training_config.get('epochs', 50) * 100,  # Estimated
            
            # Optimization
            'optimizer': self.training_config.get('optimizer', 'adam'),
            'momentum': self.training_config.get('momentum', 0.9),
            'weight_decay': self.training_config.get('weight_decay', 1e-4),
            'gradient_clipping': self.training_config.get('gradient_clipping', 1.0),
            
            # Batch processing
            'batch_size': self.training_config.get('batch_size', 2),
            'gradient_accumulation_steps': self.training_config.get('gradient_accumulation_steps', 8),
            
            # Augmentation (conservative for scientific accuracy)
            'use_augmentation': True,
            'augmentation_config': {
                'horizontal_flip': True,
                'vertical_flip': False,  # Flowers typically have orientation
                'rotation_range': 10,    # Small rotation only
                'brightness_range': 0.1,
                'contrast_range': 0.1,
                'saturation_range': 0.1,
                'hue_range': 0.05
            },
            
            # Regularization
            'dropout_rate': 0.5,
            'batch_norm_momentum': 0.99,
            'label_smoothing': 0.0,  # No smoothing for binary classification
            
            # CPU optimizations
            'mixed_precision': False,
            'xla_compile': False,  # Can be slower on CPU
            'deterministic': True   # For reproducibility
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation configuration following scientific standards.
        
        Returns:
            Dictionary with evaluation parameters
        """
        eval_config = self.config.get('evaluation', {})
        
        return {
            # Detection thresholds
            'confidence_threshold': eval_config.get('confidence_threshold', 0.5),
            'iou_threshold': eval_config.get('iou_threshold', 0.5),
            'max_detections': eval_config.get('max_detections', 100),
            
            # Success criteria (from architecture documents)
            'precision_threshold': eval_config.get('precision_threshold', 0.98),
            'recall_threshold': eval_config.get('recall_threshold', 0.85),
            'f1_threshold': eval_config.get('f1_threshold', 0.91),
            
            # COCO-style evaluation
            'use_coco_metrics': True,
            'iou_thresholds': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            'area_ranges': {
                'all': [0, float('inf')],
                'small': [0, 32**2],
                'medium': [32**2, 96**2],
                'large': [96**2, float('inf')]
            },
            
            # Scientific metrics
            'detailed_metrics': eval_config.get('detailed_metrics', True),
            'save_predictions': eval_config.get('save_predictions', True),
            'save_confusion_matrix': eval_config.get('save_confusion_matrix', True),
            
            # Challenge set evaluation (Decision A2)
            'challenge_set_size': eval_config.get('challenge_set_size', 100),
            'challenge_evaluation_frequency': eval_config.get('challenge_evaluation_frequency', 5)
        }


class ModelFactory:
    """
    Factory class for creating Mask R-CNN models with different configurations.
    
    This class provides a clean interface for model creation with CPU optimizations
    and scientific reproducibility built in.
    """
    
    def __init__(self, config):
        """
        Initialize model factory.
        
        Args:
            config: Main configuration object
        """
        self.config = config
        self.model_config = MaskRCNNConfig(config)
        
    def create_model(self, mode: str = 'training') -> tf.keras.Model:
        """
        Create Mask R-CNN model for specified mode.
        
        Args:
            mode: Model mode ('training', 'inference', 'export')
            
        Returns:
            Configured TensorFlow Keras model
            
        Raises:
            ValueError: If mode is not supported
        """
        if mode not in ['training', 'inference', 'export']:
            raise ValueError(f"Unsupported mode: {mode}")
        
        logger.info(f"Creating Mask R-CNN model for {mode} mode")
        
        # Set CPU optimizations
        self._configure_tensorflow_for_cpu()
        
        # Create model based on mode
        if mode == 'training':
            model = self._create_training_model()
        elif mode == 'inference':
            model = self._create_inference_model()
        else:  # export
            model = self._create_export_model()
        
        logger.info(f"Model created successfully: {model.summary()}")
        return model
    
    def _configure_tensorflow_for_cpu(self):
        """Configure TensorFlow for optimal CPU performance."""
        # CPU thread configuration
        intraop_threads = self.config.get('hardware.intraop_threads', 8)
        interop_threads = self.config.get('hardware.interop_threads', 4)
        
        tf.config.threading.set_inter_op_parallelism_threads(interop_threads)
        tf.config.threading.set_intra_op_parallelism_threads(intraop_threads)
        
        # Disable GPU
        tf.config.set_visible_devices([], 'GPU')
        
        # Memory optimization
        tf.config.optimizer.set_jit(False)  # XLA can be slower on CPU
        
        logger.info(f"TensorFlow configured for CPU: "
                   f"intraop={intraop_threads}, interop={interop_threads}")
    
    def _create_training_model(self) -> tf.keras.Model:
        """
        Create model optimized for training.
        
        Returns:
            Training-optimized Mask R-CNN model
        """
        # This is a simplified implementation
        # In a full implementation, you would use TensorFlow Object Detection API
        # or implement the full Mask R-CNN architecture
        
        input_shape = (*self.model_config.input_size, 3)
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape, name='input_image')
        
        # Backbone (simplified ResNet-50 feature extractor)
        backbone_config = self.model_config.get_backbone_config()
        
        if backbone_config['architecture'] == 'ResNet50':
            backbone = tf.keras.applications.ResNet50(
                weights=backbone_config['weights'],
                include_top=False,
                input_tensor=inputs
            )
            
            # Freeze early layers for stability
            for layer in backbone.layers:
                if layer.name in backbone_config['layers_to_freeze']:
                    layer.trainable = False
            
            # Extract feature maps for FPN
            feature_maps = self._extract_feature_maps(backbone, backbone_config['feature_layers'])
        
        else:
            raise NotImplementedError(f"Backbone {backbone_config['architecture']} not implemented")
        
        # Feature Pyramid Network
        fpn_features = self._build_fpn(feature_maps)
        
        # Region Proposal Network
        rpn_outputs = self._build_rpn(fpn_features)
        
        # ROI heads
        roi_outputs = self._build_roi_heads(fpn_features, rpn_outputs)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=roi_outputs, name='mask_rcnn_training')
        
        # Compile model
        self._compile_model(model, mode='training')
        
        return model
    
    def _create_inference_model(self) -> tf.keras.Model:
        """
        Create model optimized for inference.
        
        Returns:
            Inference-optimized Mask R-CNN model
        """
        # Simplified inference model
        # Would implement post-processing, NMS, etc.
        training_model = self._create_training_model()
        
        # Add post-processing layers
        # This would include NMS, confidence filtering, etc.
        
        return training_model  # Simplified for now
    
    def _create_export_model(self) -> tf.keras.Model:
        """
        Create model for export/serving.
        
        Returns:
            Export-ready Mask R-CNN model
        """
        inference_model = self._create_inference_model()
        
        # Add serving signature if needed
        # Convert to TensorFlow Lite if required
        
        return inference_model  # Simplified for now
    
    def _extract_feature_maps(self, backbone: tf.keras.Model, 
                             feature_layers: List[str]) -> Dict[str, tf.Tensor]:
        """
        Extract feature maps from backbone network.
        
        Args:
            backbone: Backbone network
            feature_layers: Names of layers to extract features from
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        feature_maps = {}
        
        for layer_name in feature_layers:
            try:
                layer = backbone.get_layer(layer_name)
                feature_maps[layer_name] = layer.output
            except ValueError:
                logger.warning(f"Layer {layer_name} not found in backbone")
        
        return feature_maps
    
    def _build_fpn(self, feature_maps: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Build Feature Pyramid Network.
        
        Args:
            feature_maps: Input feature maps from backbone
            
        Returns:
            FPN feature maps
        """
        fpn_config = self.model_config.get_fpn_config()
        
        # Simplified FPN implementation
        # In full implementation, would create proper top-down pathway
        
        fpn_features = {}
        num_filters = fpn_config['num_filters']
        
        for name, feature_map in feature_maps.items():
            # Simple 1x1 conv to normalize channel dimensions
            fpn_feature = tf.keras.layers.Conv2D(
                num_filters, 1, padding='same', 
                activation='relu', name=f'fpn_{name}'
            )(feature_map)
            
            fpn_features[name] = fpn_feature
        
        return fpn_features
    
    def _build_rpn(self, fpn_features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Build Region Proposal Network.
        
        Args:
            fpn_features: FPN feature maps
            
        Returns:
            RPN outputs (objectness, bbox regression)
        """
        rpn_config = self.model_config.get_rpn_config()
        
        # Simplified RPN implementation
        rpn_outputs = {}
        
        for name, feature_map in fpn_features.items():
            # Objectness classification
            objectness = tf.keras.layers.Conv2D(
                len(rpn_config['anchor_ratios']), 3, padding='same',
                activation='sigmoid', name=f'rpn_objectness_{name}'
            )(feature_map)
            
            # Bounding box regression
            bbox_regression = tf.keras.layers.Conv2D(
                len(rpn_config['anchor_ratios']) * 4, 3, padding='same',
                name=f'rpn_bbox_{name}'
            )(feature_map)
            
            rpn_outputs[f'{name}_objectness'] = objectness
            rpn_outputs[f'{name}_bbox'] = bbox_regression
        
        return rpn_outputs
    
    def _build_roi_heads(self, fpn_features: Dict[str, tf.Tensor],
                        rpn_outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Build ROI classification and mask heads.
        
        Args:
            fpn_features: FPN feature maps
            rpn_outputs: RPN outputs
            
        Returns:
            ROI head outputs
        """
        roi_config = self.model_config.get_roi_config()
        
        # Simplified ROI heads implementation
        # In full implementation, would include ROI pooling, etc.
        
        # Global average pooling for simplified implementation
        pooled_features = []
        for feature_map in fpn_features.values():
            pooled = tf.keras.layers.GlobalAveragePooling2D()(feature_map)
            pooled_features.append(pooled)
        
        # Concatenate features
        pooled_features = [p for p in pooled_features if p is not None]
        if not pooled_features:
            raise ValueError("No valid FPN features extracted. Check backbone layer names.")
        if len(pooled_features) == 1:
            combined_features = pooled_features[0]
        else:
            combined_features = tf.keras.layers.Concatenate()(pooled_features)
        
        # Classification head
        for units in roi_config['fc_layers']:
            combined_features = tf.keras.layers.Dense(
                units, activation='relu'
            )(combined_features)
            combined_features = tf.keras.layers.Dropout(
                roi_config['dropout_rate']
            )(combined_features)
        
        # Final classification
        num_classes = self.model_config.num_classes
        classification = tf.keras.layers.Dense(
            num_classes, activation='softmax', name='classification'
        )(combined_features)
        
        # Bounding box regression
        bbox_regression = tf.keras.layers.Dense(
            num_classes * 4, name='bbox_regression'
        )(combined_features)
        
        # Simplified mask prediction
        mask_prediction = tf.keras.layers.Dense(
            num_classes * roi_config['mask_resolution']**2,
            activation='sigmoid', name='mask_prediction'
        )(combined_features)
        
        return {
            'classification': classification,
            'bbox_regression': bbox_regression,
            'mask_prediction': mask_prediction
        }
    
    def _compile_model(self, model: tf.keras.Model, mode: str):
        """
        Compile model with appropriate loss functions and optimizer.
        
        Args:
            model: Model to compile
            mode: Compilation mode
        """
        training_config = self.model_config.get_training_config()
        
        # Optimizer
        if training_config['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'] == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=training_config['learning_rate'],
                momentum=training_config['momentum'],
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
        
        # Loss functions
        losses = {
            'classification': 'categorical_crossentropy',
            'bbox_regression': 'huber',  # Robust to outliers
            'mask_prediction': 'binary_crossentropy'
        }
        
        # Loss weights (from architecture documents)
        loss_weights = {
            'classification': 1.0,
            'bbox_regression': 1.0,
            'mask_prediction': 1.0
        }
        
        # Metrics
        metrics = {
            'classification': ['accuracy', 'precision', 'recall'],
            'bbox_regression': ['mae'],
            'mask_prediction': ['binary_accuracy']
        }
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        logger.info(f"Model compiled for {mode} with optimizer: {training_config['optimizer']}")
    
    def load_pretrained_weights(self, model: tf.keras.Model, 
                               weights_path: Optional[Path] = None) -> tf.keras.Model:
        """
        Load pre-trained weights into model.
        
        Args:
            model: Model to load weights into
            weights_path: Optional path to weights file
            
        Returns:
            Model with loaded weights
        """
        if weights_path and weights_path.exists():
            try:
                model.load_weights(str(weights_path))
                logger.info(f"Loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights from {weights_path}: {e}")
        
        return model
    
    def save_model_config(self, output_path: Path):
        """
        Save model configuration for reproducibility.
        
        Args:
            output_path: Path to save configuration
        """
        config_dict = {
            'backbone': self.model_config.get_backbone_config(),
            'rpn': self.model_config.get_rpn_config(),
            'roi': self.model_config.get_roi_config(),
            'fpn': self.model_config.get_fpn_config(),
            'training': self.model_config.get_training_config(),
            'evaluation': self.model_config.get_evaluation_config()
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model configuration saved to {output_path}")
