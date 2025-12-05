"""
Configuration Manager

Handles loading and managing configuration from YAML files.
Provides adaptive configuration based on video properties.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and access."""
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to custom config file. If None, uses default.
        """
        self.config: Dict[str, Any] = {}
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        # Load default config first
        if self.DEFAULT_CONFIG_PATH.exists():
            with open(self.DEFAULT_CONFIG_PATH, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Override with custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self._merge_config(custom_config)
    
    def _merge_config(self, custom_config: Dict[str, Any]) -> None:
        """Recursively merge custom config into default config."""
        def merge(base: dict, override: dict) -> dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merge(self.config, custom_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'court_detection.method')
            default: Default value if key not found
            
        Returns:
            Configuration value
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
            key: Configuration key (e.g., 'court_detection.method')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_for_video(self, width: int, height: int, fps: float) -> None:
        """
        Update configuration based on video properties.
        Makes parameters adaptive to video dimensions.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Video frames per second
        """
        diagonal = (width ** 2 + height ** 2) ** 0.5
        
        # Update edge detection parameters
        min_line_length = int(diagonal * self.get('court_detection.edge.min_line_length_ratio', 0.1))
        max_line_gap = int(diagonal * self.get('court_detection.edge.max_line_gap_ratio', 0.02))
        self.set('court_detection.edge.min_line_length', min_line_length)
        self.set('court_detection.edge.max_line_gap', max_line_gap)
        
        # Update feather radius
        feather_radius = int(min(width, height) * self.get('court_masking.feather_radius_ratio', 0.005))
        self.set('court_masking.feather_radius', max(feather_radius, 1))
        
        # Update trajectory length based on fps
        trajectory_seconds = 1.0  # 1 second of trajectory
        trajectory_length = int(fps * trajectory_seconds)
        self.set('player_tracking.visualization.trajectory_length', trajectory_length)
        
        # Store video properties
        self.set('video.width', width)
        self.set('video.height', height)
        self.set('video.detected_fps', fps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def save(self, path: str) -> None:
        """Save current configuration to file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create global configuration instance.
    
    Args:
        config_path: Path to custom config file
        
    Returns:
        ConfigManager instance
    """
    global _config_instance
    
    if _config_instance is None or config_path is not None:
        _config_instance = ConfigManager(config_path)
    
    return _config_instance
