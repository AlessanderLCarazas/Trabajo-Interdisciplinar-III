"""
CARGADOR Y PROCESADOR DE DATOS - BookSum Dataset
==================================================
Este archivo maneja toda la carga y preprocesamiento del dataset BookSum.

QUÉ HACE:
- Descarga y procesa el dataset BookSum de HuggingFace
- Construye el vocabulario (word2idx, idx2word) y lo guarda en data/vocab.pkl
- Tokeniza documentos y resúmenes en oraciones y palabras
- Crea resúmenes "oracle" (oraciones que mejor aproximan el gold summary)
- Prepara batches con padding para entrenamiento

CLASES PRINCIPALES:
- BookSumDataset: Dataset personalizado para BookSum
- collate_fn(): Función para crear batches con padding
- create_data_loaders(): Crea train/val/test loaders

PROCESAMIENTO:
1. Carga capítulos de libros y sus resúmenes
2. Filtra por longitud (min/max oraciones)
3. Tokeniza con NLTK
4. Crea vocabulario de palabras más frecuentes
5. Genera oracle summaries usando ROUGE para training

CONFIGURACIÓN:
- max_train_docs, max_val_docs: límites para runs rápidos
- max_doc_len, max_sent_len: límites de tokenización
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm
import logging

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class BookSumDataset(Dataset):
    def __init__(self, config, split='train', max_docs=None):
        self.config = config
        self.split = split
        self.max_doc_len = config.get('model.max_doc_len', 500)
        self.max_sent_len = config.get('model.max_sent_len', 100)
        self.max_summary_length = config.get('data.max_summary_length', 10)
        self.min_summary_length = config.get('data.min_summary_length', 3)
        
        # Load vocabulary
        self.vocab_path = os.path.join(config.get('paths.data_dir', './data'), 'vocab.pkl')
        self.word2idx, self.idx2word = self._load_or_create_vocab()
        
        # Load dataset
        self.data = self._load_booksum_data(split, max_docs)
        
        logger.info(f"Loaded {len(self.data)} examples for {split} split")
    
    def _load_booksum_data(self, split, max_docs=None):
        """Load BookSum dataset from HuggingFace"""
        try:
            # Load BookSum dataset
            dataset = load_dataset("kmfoda/booksum", split=split)
            
            processed_data = []
            for i, example in enumerate(tqdm(dataset, desc=f"Processing {split} data")):
                if max_docs and i >= max_docs:
                    break
                
                # Get text and summary
                text = example.get('chapter', '') or example.get('text', '')
                summary = example.get('summary_text', '') or example.get('summary', '')
                
                if not text or not summary:
                    continue
                
                # Tokenize into sentences
                sentences = sent_tokenize(text)
                summary_sentences = sent_tokenize(summary)
                
                # Filter by length
                if len(sentences) < 10 or len(sentences) > self.max_doc_len:
                    continue
                
                if len(summary_sentences) < self.min_summary_length or len(summary_sentences) > self.max_summary_length:
                    continue
                
                # Tokenize sentences into words
                tokenized_sentences = []
                for sent in sentences:
                    words = word_tokenize(sent.lower())
                    if len(words) > 0 and len(words) <= self.max_sent_len:
                        tokenized_sentences.append(words)
                
                tokenized_summary = []
                for sent in summary_sentences:
                    words = word_tokenize(sent.lower())
                    if len(words) > 0:
                        tokenized_summary.append(words)
                
                if len(tokenized_sentences) >= 10 and len(tokenized_summary) >= self.min_summary_length:
                    processed_data.append({
                        'sentences': tokenized_sentences,
                        'summary': tokenized_summary,
                        'raw_text': sentences,
                        'raw_summary': summary_sentences
                    })
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Error loading BookSum dataset: {e}")
            # Fallback to dummy data for testing
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing when BookSum is not available"""
        logger.warning("Creating dummy data for testing")
        dummy_data = []
        
        for i in range(100):
            sentences = [
                ['this', 'is', 'sentence', str(j)] + ['word'] * (5 + j % 10)
                for j in range(20 + i % 30)
            ]
            summary = [
                ['summary', 'sentence', str(j)] + ['word'] * (3 + j % 5)
                for j in range(3 + i % 5)
            ]
            
            dummy_data.append({
                'sentences': sentences,
                'summary': summary,
                'raw_text': [' '.join(sent) for sent in sentences],
                'raw_summary': [' '.join(sent) for sent in summary]
            })
        
        return dummy_data
    
    def _load_or_create_vocab(self):
        """Load existing vocabulary or create new one"""
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
            return vocab_data['word2idx'], vocab_data['idx2word']
        else:
            return self._create_vocab()
    
    def _create_vocab(self):
        """Create vocabulary from all data splits"""
        logger.info("Creating vocabulary...")
        
        # Special tokens
        word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        
        word_freq = defaultdict(int)
        
        # Load all splits to build comprehensive vocabulary
        for split in ['train', 'validation', 'test']:
            try:
                dataset = load_dataset("kmfoda/booksum", split=split)
                for example in tqdm(dataset, desc=f"Building vocab from {split}"):
                    text = example.get('chapter', '') or example.get('text', '')
                    summary = example.get('summary_text', '') or example.get('summary', '')
                    
                    # Tokenize and count words
                    for sent in sent_tokenize(text):
                        for word in word_tokenize(sent.lower()):
                            word_freq[word] += 1
                    
                    for sent in sent_tokenize(summary):
                        for word in word_tokenize(sent.lower()):
                            word_freq[word] += 1
            except:
                logger.warning(f"Could not load {split} split for vocabulary building")
        
        # Add words with frequency > 2 (lowered threshold)
        for word, freq in word_freq.items():
            if freq > 2 and word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx
                idx2word[idx] = word
        
        # Ensure minimum vocabulary size
        if len(word2idx) < 1000:
            logger.warning(f"Small vocabulary size: {len(word2idx)}. Adding common words.")
            common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                           'a', 'an', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 
                           'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                           'can', 'may', 'might', 'must', 'shall', 'not', 'no', 'yes', 'all', 'some', 'many']
            
            for word in common_words:
                if word not in word2idx:
                    idx = len(word2idx)
                    word2idx[word] = idx
                    idx2word[idx] = word
        
        # Save vocabulary
        os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
        with open(self.vocab_path, 'wb') as f:
            pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
        
        logger.info(f"Created vocabulary with {len(word2idx)} words")
        return word2idx, idx2word
    
    def _encode_sentences(self, sentences):
        """Encode sentences to indices"""
        encoded = []
        for sent in sentences:
            encoded_sent = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sent]
            encoded.append(encoded_sent)
        return encoded
    
    def _create_oracle_summary(self, sentences, summary):
        """Create oracle extractive summary using greedy selection"""
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        selected_indices = []
        remaining_indices = list(range(len(sentences)))
        target_summary = ' '.join([' '.join(sent) for sent in summary])
        
        for _ in range(min(len(summary), len(sentences))):
            best_score = -1
            best_idx = -1
            
            for idx in remaining_indices:
                current_summary = ' '.join([
                    ' '.join(sentences[i]) 
                    for i in selected_indices + [idx]
                ])
                
                scores = scorer.score(target_summary, current_summary)
                rouge_score = scores['rougeL'].fmeasure
                
                if rouge_score > best_score:
                    best_score = rouge_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return sorted(selected_indices)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode sentences
        sentences = self._encode_sentences(item['sentences'])
        summary = self._encode_sentences(item['summary'])
        
        # Create oracle extractive summary
        oracle_indices = self._create_oracle_summary(item['sentences'], item['summary'])
        
        return {
            'sentences': sentences,
            'summary': summary,
            'oracle_indices': oracle_indices,
            'raw_sentences': item['raw_text'],
            'raw_summary': item['raw_summary'],
            'num_sentences': len(sentences)
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    max_doc_len = max(item['num_sentences'] for item in batch)
    max_sent_len = max(
        max(len(sent) for sent in item['sentences']) if item['sentences'] else 0
        for item in batch
    )
    
    batch_sentences = []
    batch_oracle_indices = []
    batch_masks = []
    batch_raw_sentences = []
    batch_raw_summaries = []
    
    for item in batch:
        # Pad sentences
        padded_sentences = []
        for sent in item['sentences']:
            padded_sent = sent + [0] * (max_sent_len - len(sent))
            padded_sentences.append(padded_sent)
        
        # Pad document
        while len(padded_sentences) < max_doc_len:
            padded_sentences.append([0] * max_sent_len)
        
        # Create mask
        mask = [1] * item['num_sentences'] + [0] * (max_doc_len - item['num_sentences'])
        
        # Pad oracle indices
        oracle_indices = item['oracle_indices'] + [-1] * (max_doc_len - len(item['oracle_indices']))
        
        batch_sentences.append(padded_sentences)
        batch_oracle_indices.append(oracle_indices)
        batch_masks.append(mask)
        batch_raw_sentences.append(item['raw_sentences'])
        batch_raw_summaries.append(item['raw_summary'])
    
    return {
        'sentences': torch.tensor(batch_sentences, dtype=torch.long),
        'oracle_indices': torch.tensor(batch_oracle_indices, dtype=torch.long),
        'masks': torch.tensor(batch_masks, dtype=torch.float),
        'raw_sentences': batch_raw_sentences,
        'raw_summaries': batch_raw_summaries
    }

def create_data_loaders(config):
    """Create train, validation, and test data loaders"""
    # Permitir límites para runs rápidos
    max_train_docs = config.get('data.max_train_docs', None)
    max_val_docs = config.get('data.max_val_docs', None)
    max_test_docs = config.get('data.max_test_docs', None)

    train_dataset = BookSumDataset(config, split='train', max_docs=max_train_docs)
    val_dataset = BookSumDataset(config, split='validation', max_docs=max_val_docs)
    test_dataset = BookSumDataset(config, split='test', max_docs=max_test_docs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training.batch_size', 8),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=config.get('data.pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training.batch_size', 8),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=config.get('data.pin_memory', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('training.batch_size', 8),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=config.get('data.pin_memory', True)
    )
    
    return train_loader, val_loader, test_loader, len(train_dataset.word2idx)