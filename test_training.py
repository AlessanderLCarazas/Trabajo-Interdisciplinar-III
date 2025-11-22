#!/usr/bin/env python3
"""
Quick Training Test - Minimal version for fast testing
"""

import os
import sys
import torch
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.model import MemSum
from src.trainer import RLTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_batch(batch_size=2, vocab_size=1000, max_doc_len=20, max_sent_len=15):
    """Create dummy batch for testing"""
    
    # Create random sentences
    sentences = torch.randint(1, vocab_size, (batch_size, max_doc_len, max_sent_len))
    
    # Create masks (some sentences are valid)
    masks = torch.ones(batch_size, max_doc_len)
    for i in range(batch_size):
        # Randomly set some sentences as padding
        num_real_sents = torch.randint(10, max_doc_len, (1,)).item()
        masks[i, num_real_sents:] = 0
    
    # Create oracle indices
    oracle_indices = []
    for i in range(batch_size):
        num_real_sents = int(masks[i].sum().item())
        num_oracle = min(5, num_real_sents // 2)
        oracle = torch.randperm(num_real_sents)[:num_oracle].sort()[0]
        # Pad to max_doc_len
        padded_oracle = torch.full((max_doc_len,), -1, dtype=torch.long)
        padded_oracle[:len(oracle)] = oracle
        oracle_indices.append(padded_oracle)
    
    oracle_indices = torch.stack(oracle_indices)
    
    # Create dummy raw text
    raw_sentences = []
    raw_summaries = []
    
    for i in range(batch_size):
        sents = [f"This is sentence {j} in document {i}." for j in range(int(masks[i].sum().item()))]
        summary = [f"Summary sentence {j} for doc {i}." for j in range(3)]
        raw_sentences.append(sents)
        raw_summaries.append(summary)
    
    return {
        'sentences': sentences,
        'masks': masks,
        'oracle_indices': oracle_indices,
        'raw_sentences': raw_sentences,
        'raw_summaries': raw_summaries
    }

def test_training_step():
    """Test a single training step with dummy data"""
    
    print("🧪 Testing MemSum training step...")
    
    # Configuration
    config = Config()
    config['training']['batch_size'] = 2
    config['model']['max_doc_len'] = 20
    config['model']['max_sent_len'] = 15
    config['data']['max_summary_length'] = 5
    
    vocab_size = 1000
    
    try:
        # Create model and trainer
        print("Creating model...")
        model = MemSum(vocab_size, config)
        trainer = RLTrainer(model, config, vocab_size)
        
        # Create dummy batch
        print("Creating dummy data...")
        batch = create_dummy_batch(
            batch_size=2,
            vocab_size=vocab_size,
            max_doc_len=config.get('model.max_doc_len', 20),
            max_sent_len=config.get('model.max_sent_len', 15)
        )
        
        # Test training step
        print("Testing training step...")
        metrics = trainer.train_step(batch)
        
        print("✅ Training step successful!")
        print(f"   Loss: {metrics['total_loss']:.4f}")
        print(f"   Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"   Value Loss: {metrics['value_loss']:.4f}")
        print(f"   Reward: {metrics['avg_reward']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_step()
    if success:
        print("\n🎉 Training test passed! Ready for full training.")
    else:
        print("\n❌ Training test failed. Check the errors above.")
    
    sys.exit(0 if success else 1)