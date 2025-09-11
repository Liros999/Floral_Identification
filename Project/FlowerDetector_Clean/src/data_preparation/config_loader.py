"""Configuration loader - simple and focused."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Simple configuration loader."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self._validate_config()
        self._resolve_paths()
        
        logger.info(f"Config loaded from {self.config_path}")
        return self.config
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        required = ['data', 'model', 'training']
        for section in required:
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")
                
        # Validate scientific requirements
        if self.config['training']['target_precision'] < 0.98:
            raise ValueError("target_precision must be ≥0.98")
        if self.config['training']['min_recall'] < 0.85:
            raise ValueError("min_recall must be ≥0.85")
            
    def _resolve_paths(self) -> None:
        """Resolve paths."""
        data_config = self.config['data']
        base_path = Path(data_config['google_drive_base_path'])
        
        self.config['data_paths'] = {
            'base': base_path,
            'positive_images': base_path / data_config['positive_images_subpath'],
            'negative_images': base_path / data_config['negative_images_subpath'],
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get data paths."""
        return self.config['data_paths']
