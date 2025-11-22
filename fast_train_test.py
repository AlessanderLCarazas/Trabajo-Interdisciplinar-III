#!/usr/bin/env python3
"""
Fast Training with Dummy Data - Quick test of full training pipeline
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

class DummyDataLoader:
    def __init__(self, batch_size=2, num_batches=50, vocab_size=1000, max_doc_len=20, max_sent_len=15):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.vocab_size = vocab_size
        self.max_doc_len = max_doc_len
        self.max_sent_len = max_sent_len
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        batch = self.create_batch()
        self.current_batch += 1
        return batch
    
    def __len__(self):
        return self.num_batches
    
    def create_batch(self):
        """Create dummy batch"""
        # Create random sentences
        sentences = torch.randint(1, self.vocab_size, (self.batch_size, self.max_doc_len, self.max_sent_len))
        
        # Create masks
        masks = torch.ones(self.batch_size, self.max_doc_len)
        for i in range(self.batch_size):
            num_real_sents = torch.randint(10, self.max_doc_len, (1,)).item()
            masks[i, num_real_sents:] = 0
        
        # Create oracle indices
        oracle_indices = []
        for i in range(self.batch_size):
            num_real_sents = int(masks[i].sum().item())
            num_oracle = min(5, max(2, num_real_sents // 3))
            oracle = torch.randperm(num_real_sents)[:num_oracle].sort()[0]
            # Pad to max_doc_len
            padded_oracle = torch.full((self.max_doc_len,), -1, dtype=torch.long)
            padded_oracle[:len(oracle)] = oracle
            oracle_indices.append(padded_oracle)
        
        oracle_indices = torch.stack(oracle_indices)
        
        # Create dummy raw text
        raw_sentences = []
        raw_summaries = []
        
        for i in range(self.batch_size):
            num_sents = int(masks[i].sum().item())
            sents = [f"This is sentence {j} in document {i} about topic X." for j in range(num_sents)]
            summary = [f"Summary sentence {j} for document {i}." for j in range(3)]
            raw_sentences.append(sents)
            raw_summaries.append(summary)
        
        return {
            'sentences': sentences,
            'masks': masks,
            'oracle_indices': oracle_indices,
            'raw_sentences': raw_sentences,
            'raw_summaries': raw_summaries
        }

def fast_training_test():
    """Test full training pipeline with dummy data"""
    
    print("🚀 Fast Training Test with Dummy Data")
    print("=" * 50)
    
    # Configuration
    config = Config()
    config['training']['batch_size'] = 2
    config['training']['num_epochs'] = 2
    config['model']['max_doc_len'] = 20
    config['model']['max_sent_len'] = 15
    config['data']['max_summary_length'] = 5
    config['training']['eval_every'] = 25  # Evaluate more frequently
    
    vocab_size = 1000
    
    try:
        # Create model and trainer
        print("Creating model and trainer...")
        model = MemSum(vocab_size, config)
        trainer = RLTrainer(model, config, vocab_size)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {trainer.device}")
        
        # Create dummy data loaders
        print("Creating dummy data loaders...")
        train_loader = DummyDataLoader(batch_size=2, num_batches=50, vocab_size=vocab_size)
        val_loader = DummyDataLoader(batch_size=2, num_batches=10, vocab_size=vocab_size)
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Test training
        print("Starting fast training test...")
        best_rouge = trainer.train(train_loader, val_loader, num_epochs=2)
        
        print(f"✅ Training completed! Best ROUGE-L: {best_rouge:.4f}")
        
        # Test evaluation
        print("Testing evaluation...")
        val_metrics = trainer.evaluate(val_loader)
        print(f"Final validation metrics: {val_metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fast training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fast_training_test()
    if success:
        print("\n🎉 Fast training test passed! Ready for BookSum training.")
        print("\nNext steps:")
        print("1. Run: ./quick_start.sh")
        print("2. Select option 2 for quick BookSum training")
        print("3. Select option 3 for full BookSum training")
    else:
        print("\n❌ Fast training test failed. Check the errors above.")
    
    sys.exit(0 if success else 1)