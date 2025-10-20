# src/config/model_config.py

import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import json

@dataclass
class ModelConfig:
    """Container for all model parameters"""
    vfa: Dict
    cfa: Dict
    dla: Dict
    pfa: Dict
    learning: Dict
    coordination: Dict
    system: Dict
    integration: Dict

class ModelConfigurator:
    """
    Manages model parameters from YAML config files
    Version controlled, requires restart to change
    """
    
    def __init__(self, config_path: str = "config/model_parameters.yaml",
                 env: str = "production"):
        self.config_path = Path(config_path)
        self.env = env
        self.config: Optional[ModelConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load and validate configuration from file"""
        # Load base config
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Load environment-specific overrides if they exist
        env_config_path = self.config_path.parent / f"environments/{self.env}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_overrides = yaml.safe_load(f)
            config_dict = self._merge_configs(config_dict, env_overrides)
        
        # Validate schema
        self._validate_config(config_dict)
        
        # Create ModelConfig object
        self.config = ModelConfig(
            vfa=config_dict['vfa'],
            cfa=config_dict['cfa'],
            dla=config_dict['dla'],
            pfa=config_dict['pfa'],
            learning=config_dict['learning'],
            coordination=config_dict['coordination'],
            system=config_dict['system'],
            integration=config_dict['integration']
        )
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge override config into base config"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self, config_dict: Dict):
        """Validate configuration against schema"""
        required_sections = ['vfa', 'cfa', 'dla', 'pfa', 'learning', 
                           'coordination', 'system', 'integration']
        
        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate VFA parameters
        vfa = config_dict['vfa']
        assert 0 < vfa['learning']['initial_learning_rate'] <= 1.0
        assert 0 < vfa['learning']['discount_factor'] < 1.0
        assert 0 <= vfa['exploration']['final_epsilon'] <= vfa['exploration']['initial_epsilon'] <= 1.0
        
        # Validate CFA parameters
        cfa = config_dict['cfa']
        assert cfa['solver']['time_limit_seconds'] > 0
        assert cfa['batching']['min_batch_size'] >= 1
        assert cfa['batching']['max_batch_size'] >= cfa['batching']['min_batch_size']
        
        # Validate DLA parameters
        dla = config_dict['dla']
        assert dla['lookahead']['horizon_hours'] > 0
        assert dla['lookahead']['num_monte_carlo_samples'] > 0
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get nested config value using dot notation
        Example: config.get('vfa.learning.learning_rate')
        """
        keys = path.split('.')
        value = self.config.__dict__
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict:
        """Get entire config section"""
        return getattr(self.config, section, {})
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
    
    def export_json(self, output_path: str):
        """Export current config to JSON for documentation"""
        config_dict = {
            'vfa': self.config.vfa,
            'cfa': self.config.cfa,
            'dla': self.config.dla,
            'pfa': self.config.pfa,
            'learning': self.config.learning,
            'coordination': self.config.coordination,
            'system': self.config.system,
            'integration': self.config.integration
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)