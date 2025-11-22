#!/usr/bin/env python3
"""
SCRIPT PRINCIPAL DE ENTRENAMIENTO - MemSum para BookSum
========================================================
Este es el archivo más importante para entrenar el modelo.

QUÉ HACE:
- Carga la configuración desde configs/booksum_config.yaml
- Prepara los datos de BookSum (train/validation/test)
- Crea el modelo MemSum y el entrenador con reinforcement learning
- Ejecuta el bucle de entrenamiento completo (épocas, validación, checkpoints)
- Guarda el mejor modelo y las métricas finales

CÓMO USARLO:
- Entrenar: python train.py --config configs/booksum_config.yaml --epochs 40 --batch_size 2
- Reanudar: python train.py --resume checkpoints/checkpoint_epoch_X.pt

SALIDAS:
- checkpoints/: modelos guardados (.pt)
- logs/: configuración y resultados finales (JSON)
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import random
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.data_loader import create_data_loaders, BookSumDataset
from src.model import MemSum
from src.trainer import RLTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce verbosity from third-party libraries (absl, transformers)
try:
    logging.getLogger('absl').setLevel(logging.WARNING)
except Exception:
    pass
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn flags are configured later based on config for performance

def check_gpu_availability():
    """Check and log GPU availability"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9
        
        logger.info(f"GPU available: {gpu_name}")
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        logger.info(f"Number of GPUs: {gpu_count}")
        logger.info(f"Current GPU: {current_gpu}")
        
        return True
    else:
        logger.warning("No GPU available, using CPU")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train MemSum on BookSum dataset')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data_limit', type=int, default=None,
                       help='Limit number of training examples (for testing)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Set random seed (before any heavy init) and configure cuDNN per config
    set_random_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    torch.backends.cudnn.benchmark = config.get('device.cudnn_benchmark', True)
    torch.backends.cudnn.deterministic = config.get('device.cudnn_deterministic', False)
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.wandb:
        config['logging']['use_wandb'] = True
    
    # Adjust config based on GPU availability
    if not gpu_available:
        config['device']['use_gpu'] = False
        config['device']['mixed_precision'] = False
        logger.info("Disabled GPU and mixed precision due to unavailability")
    
    logger.info(f"Configuration loaded: {config.config}")
    
    # Create directories
    os.makedirs(config.get('paths.data_dir', './data'), exist_ok=True)
    os.makedirs(config.get('paths.model_dir', './models'), exist_ok=True)
    os.makedirs(config.get('paths.checkpoint_dir', './checkpoints'), exist_ok=True)
    os.makedirs(config.get('paths.log_dir', './logs'), exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(config.get('paths.log_dir', './logs'), 'config.yaml')
    config.save(config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    try:
        # Create data loaders
        logger.info("Loading BookSum dataset...")
        train_loader, val_loader, test_loader, vocab_size = create_data_loaders(config)
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Vocabulary size: {vocab_size}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Create model
        logger.info("Creating MemSum model...")
        model = MemSum(vocab_size, config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = RLTrainer(model, config, vocab_size)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            if os.path.exists(args.resume):
                start_epoch = trainer.load_checkpoint(args.resume)
                logger.info(f"Resumed training from epoch {start_epoch}")
            else:
                logger.warning(f"Checkpoint not found: {args.resume}")
        
        # Training
        logger.info("Starting training...")
        num_epochs = config.get('training.num_epochs', 20)
        
        best_rouge = trainer.train(train_loader, val_loader, num_epochs)
        
        logger.info(f"Training completed! Best ROUGE-L: {best_rouge:.4f}")
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test set ROUGE scores: {test_metrics}")
        
        # Save final results
        # Asegurar que la config sea serializable (por ejemplo, torch.device)
        import json
        def to_serializable(obj):
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return str(obj)

        serializable_config = json.loads(json.dumps(config.config, default=to_serializable))

        results = {
            'best_validation_rouge': best_rouge,
            'test_rouge_scores': test_metrics,
            'config': serializable_config,
            'model_parameters': {
                'total': total_params,
                'trainable': trainable_params
            }
        }
        
        results_path = os.path.join(config.get('paths.log_dir', './logs'), 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {results_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    logger.info("Training script completed successfully!")

if __name__ == '__main__':
    main()