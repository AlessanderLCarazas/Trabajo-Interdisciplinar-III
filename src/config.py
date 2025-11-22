import os
import torch
from omegaconf import DictConfig
import yaml

# Default configuration for MemSum
DEFAULT_CONFIG = {
    'model': {
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'word_embed_dim': 200,
        'pos_embed_dim': 50,
        'sent_embed_dim': 256,
        'doc_embed_dim': 256,
        'memory_dim': 256,
        'max_doc_len': 500,
        'max_sent_len': 100,
    },
    'training': {
        'learning_rate': 1e-4,
        'batch_size': 8,
        'num_epochs': 20,
        'warmup_steps': 1000,
        'gradient_clip_norm': 5.0,
        'accumulation_steps': 4,
        'eval_every': 1000,
        'save_every': 2000,
        'early_stopping_patience': 5,
    },
    'rl': {
        'gamma': 0.95,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_episodes': 100000,
        'memory_size': 50000,
        'exploration_noise': 0.1,
    },
    'data': {
        'dataset_name': 'booksum',
        'max_summary_length': 10,
        'min_summary_length': 3,
        'train_split': 'train',
        'val_split': 'validation',
        'test_split': 'test',
        'num_workers': 4,
        'pin_memory': True,
    },
    'paths': {
        'data_dir': './data',
        'model_dir': './models',
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints',
    },
    'device': {
        'use_gpu': True,
        'gpu_id': 0,
        'mixed_precision': True,
    },
    'logging': {
        'use_wandb': False,
        'wandb_project': 'memsum-booksum',
        'log_level': 'INFO',
    }
}

class Config:
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.config = self._merge_configs(DEFAULT_CONFIG, config)
        else:
            self.config = DEFAULT_CONFIG.copy()
        
        # Set device
        if self.config['device']['use_gpu'] and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config['device']['gpu_id']}")
        else:
            self.device = torch.device('cpu')
        
        self.config['device']['device'] = self.device
    
    def _merge_configs(self, default, custom):
        """Recursively merge custom config with default config"""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        # Convert string representations of numbers to actual numbers
        if isinstance(value, str):
            # Try to convert scientific notation or decimal strings to float
            try:
                if 'e-' in value.lower() or 'e+' in value.lower():
                    return float(value)
                elif '.' in value:
                    return float(value)
                elif value.isdigit():
                    return int(value)
            except ValueError:
                pass
        
        return value
    
    def save(self, path):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value