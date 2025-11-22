#!/usr/bin/env python3
"""
Simple Test Script - Quick verification that MemSum is working
"""

import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def quick_test():
    print("🚀 MemSum Quick Test")
    print("=" * 30)
    
    try:
        # Test 1: GPU
        print("1. GPU Test...")
        if torch.cuda.is_available():
            print(f"   ✅ GPU: {torch.cuda.get_device_name()}")
            print(f"   ✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️ No GPU (will use CPU)")
        
        # Test 2: Config
        print("2. Configuration Test...")
        from src.config import Config
        config = Config("configs/booksum_config.yaml")
        lr = config.get('training.learning_rate')
        print(f"   ✅ Learning rate: {lr} (type: {type(lr)})")
        
        # Test 3: Model Creation
        print("3. Model Creation Test...")
        from src.model import MemSum
        vocab_size = 1000
        model = MemSum(vocab_size, config)
        params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model created: {params:,} parameters")
        
        # Test 4: Forward Pass
        print("4. Forward Pass Test...")
        batch_size = 1
        doc_len = 10
        sent_len = 20
        
        sentences = torch.randint(0, vocab_size, (batch_size, doc_len, sent_len))
        mask = torch.ones(batch_size, doc_len)
        
        with torch.no_grad():
            outputs = model(sentences, mask)
        print(f"   ✅ Forward pass successful")
        
        # Test 5: Training Setup
        print("5. Training Setup Test...")
        from src.trainer import RLTrainer
        trainer = RLTrainer(model, config, vocab_size)
        print(f"   ✅ Trainer created on device: {trainer.device}")
        
        print("\n🎉 All tests passed! MemSum is ready to train.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)