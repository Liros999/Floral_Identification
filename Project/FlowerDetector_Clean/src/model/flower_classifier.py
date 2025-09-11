"""Simplified EfficientNet-B0 flower classifier - optimized for binary classification."""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class FlowerClassifier(nn.Module):
    """
    Simplified EfficientNet-B0-based binary classifier.
    
    Architecture: EfficientNet-B0 backbone + 3-layer MLP classifier
    - ~5.3M parameters (efficient and appropriate for dataset size)
    - No attention mechanism (unnecessary for binary classification)
    - Optimized for flower vs background classification
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.3):
        super(FlowerClassifier, self).__init__()
        
        if num_classes != 2:
            raise ValueError("Binary classifier - num_classes must be 2")
        
        # Load pretrained EfficientNet-B0 using modern weights API
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Get feature dimensions
        backbone_features = self.backbone.classifier[1].in_features
        
        # Remove final classifier
        self.backbone.classifier = nn.Identity()
        
        # Remove attention mechanism - it's causing over-parameterization
        # Standard EfficientNet features are sufficient for binary classification
        
        # Optimized classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"SimplifiedFlowerClassifier initialized with EfficientNet-B0")
        logger.info(f"Parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"Architecture: EfficientNet-B0 + 3-layer MLP (no attention)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified forward pass - EfficientNet backbone + classifier."""
        # Backbone features (EfficientNet-B0 produces excellent global features)
        features = self.backbone(x)  # [batch, 1280]
        
        # Direct classification (no attention needed for binary classification)
        logits = self.classifier(features)  # [batch, 2]
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions."""
        probabilities = self.predict_proba(x)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions

def create_flower_classifier(config: Dict[str, Any]) -> FlowerClassifier:
    """Create classifier from config."""
    model_config = config.get('model', {})
    
    model = FlowerClassifier(
        num_classes=model_config.get('num_classes', 2),
        pretrained=model_config.get('pretrained', True),
        dropout_rate=model_config.get('dropout_rate', 0.5)
    )
    
    # Optional: freeze a number of backbone modules if configured
    try:
        layers_to_freeze = int(model_config.get('freeze_backbone_layers', 0) or 0)
        if layers_to_freeze > 0:
            frozen = 0
            for module in model.backbone.children():
                for param in module.parameters():
                    param.requires_grad = False
                frozen += 1
                if frozen >= layers_to_freeze:
                    break
            logger.info(f"Applied freezing to first {frozen} backbone modules")
    except Exception as e:
        logger.warning(f"Could not apply backbone freezing: {e}")
    
    # Set to CPU
    device = config.get('training', {}).get('device', 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"Created FlowerClassifier on {device}")
    return model
