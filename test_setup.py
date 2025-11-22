#!/usr/bin/env python3
"""
Test Script for MemSum Implementation
Quick test to verify installation and GPU setup
"""

import os
import sys
import torch
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_gpu():
    """Test GPU availability and configuration"""
    print("=== GPU Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        
        # Memory info
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
        
        # Test tensor operations
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✅ GPU tensor operations working!")
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
    else:
        print("❌ No GPU available")

def test_imports():
    """Test all required imports"""
    print("\n=== Import Test ===")
    
    required_modules = [
        'torch',
        'transformers',
        'datasets',
        'nltk',
        'rouge_score',
        'tqdm',
        'numpy',
        'pandas'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")

def test_config():
    """Test configuration loading"""
    print("\n=== Configuration Test ===")
    
    try:
        from src.config import Config
        
        # Test default config
        config = Config()
        print("✅ Default config loaded")
        
        # Test config file
        config_path = "configs/booksum_config.yaml"
        if os.path.exists(config_path):
            config = Config(config_path)
            print("✅ YAML config loaded")
        else:
            print("⚠️ YAML config not found, using defaults")
        
        # Test device detection
        device = config.get('device.device', torch.device('cpu'))
        print(f"✅ Device configured: {device}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def test_model_creation():
    """Test model instantiation"""
    print("\n=== Model Test ===")
    
    try:
        from src.config import Config
        from src.model import MemSum
        
        config = Config()
        vocab_size = 1000
        
        model = MemSum(vocab_size, config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        
        # Test forward pass
        batch_size = 2
        max_doc_len = 10
        max_sent_len = 20
        
        sentences = torch.randint(0, vocab_size, (batch_size, max_doc_len, max_sent_len))
        mask = torch.ones(batch_size, max_doc_len)
        
        with torch.no_grad():
            outputs = model(sentences, mask)
        
        print(f"✅ Forward pass successful")
        print(f"   Action logits shape: {outputs['action_logits'].shape}")
        print(f"   Values shape: {outputs['values'].shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

def test_data_loader():
    """Test data loading (with dummy data)"""
    print("\n=== Data Loading Test ===")
    
    try:
        from src.config import Config
        from src.data_loader import BookSumDataset
        
        config = Config()
        
        # Create small dataset for testing
        dataset = BookSumDataset(config, split='train', max_docs=5)
        print(f"✅ Dataset created with {len(dataset)} examples")
        
        # Test getting an item
        item = dataset[0]
        print(f"✅ Data item retrieved")
        print(f"   Sentences: {len(item['sentences'])}")
        print(f"   Oracle indices: {len(item['oracle_indices'])}")
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        print("   This is expected if BookSum dataset is not available")
        print("   The training script will create dummy data automatically")

def test_training_setup():
    """Test training setup without actually training"""
    print("\n=== Training Setup Test ===")
    
    try:
        from src.config import Config
        from src.model import MemSum
        from src.trainer import RLTrainer
        
        config = Config()
        vocab_size = 1000
        
        model = MemSum(vocab_size, config)
        trainer = RLTrainer(model, config, vocab_size)
        
        print("✅ Trainer created successfully")
        print(f"   Device: {trainer.device}")
        print(f"   Mixed precision: {trainer.use_amp}")
        print(f"   Learning rate: {config.get('training.learning_rate', 'default')}")
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")

def main():
    """Run all tests"""
    print("🧪 MemSum Implementation Test Suite")
    print("=" * 50)
    
    test_gpu()
    test_imports()
    test_config()
    test_model_creation()
    test_data_loader()
    test_training_setup()
    
    print("\n" + "=" * 50)
    print("🎉 Test suite completed!")
    print("\nNext steps:")
    print("1. Run: python train.py --config configs/booksum_config.yaml --epochs 1")
    print("2. Monitor GPU usage: nvidia-smi")
    print("3. Check logs: tail -f training.log")

if __name__ == "__main__":
    main()