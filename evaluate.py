#!/usr/bin/env python3
"""
EVALUACIÓN E INFERENCIA - Script para usar el modelo entrenado
================================================================
Este script permite evaluar el modelo y generar resúmenes de textos nuevos.

QUÉ HACE:
- Carga un modelo entrenado desde checkpoints/
- Evalúa métricas ROUGE, BERTScore y SummaQA en test set
- Genera resúmenes extractivos de cualquier texto
- Soporte para inferencia interactiva

CLASES PRINCIPALES:
- MemSumInference: Clase para generar resúmenes de textos arbitrarios
- evaluate_model(): Función para evaluar en dataset completo

USO TÍPICO:
- Evaluar en test: python evaluate.py checkpoints/best_model.pt --split test
- Resumir texto: python evaluate.py checkpoints/best_model.pt --text "Su texto aquí..."
- Guardar resultados: python evaluate.py checkpoints/best_model.pt --output resultados.json

CARACTERÍSTICAS:
- Carga vocabulario automáticamente desde data/vocab.pkl
- Fallback a CPU si GPU no disponible
- Soporte para múltiples idiomas en tokenización
- Salida en formato JSON con métricas detalladas

MÉTRICAS IMPLEMENTADAS:
- ROUGE-1, ROUGE-2, ROUGE-L: Solapamiento léxico (unigramas, bigramas, LCS)
- BERTScore: Similitud semántica basada en embeddings de BERT
- SummaQA: Evaluación basada en QA (content coverage)
"""

import os
import sys
import argparse
import logging
import torch
import json
from pathlib import Path
from rouge_score import rouge_scorer
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.data_loader import create_data_loaders, BookSumDataset
from src.model import MemSum
from src.trainer import RLTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemSumInference:
    """MemSum inference class for generating summaries"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        self.config = Config(config_path)
        # Device fallback: if config says cuda but no GPU available, use CPU
        cfg_device = self.config.get('device.device', torch.device('cpu'))
        try:
            if isinstance(cfg_device, str):
                cfg_device = torch.device(cfg_device)
        except Exception:
            cfg_device = torch.device('cpu')

        if getattr(cfg_device, 'type', 'cpu') == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = cfg_device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load vocabulary from saved vocab.pkl
        self.word2idx, self.idx2word = self._load_vocab()
        vocab_size = len(self.word2idx) if self.word2idx else 10000
        
        # Create model
        self.model = MemSum(vocab_size, self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        try:
            self.model.to(self.device)
        except Exception:
            # Last-resort fallback to CPU
            self.device = torch.device('cpu')
            self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {checkpoint_path}")

    def _load_vocab(self):
        """Load vocabulary produced during training (data/vocab.pkl)"""
        import pickle
        data_dir = self.config.get('paths.data_dir', './data')
        vocab_path = os.path.join(data_dir, 'vocab.pkl')
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, 'rb') as f:
                    vocab = pickle.load(f)
                return vocab.get('word2idx', {}), vocab.get('idx2word', {})
            except Exception as e:
                logger.warning(f"Failed to load vocabulary from {vocab_path}: {e}")
        else:
            logger.warning(f"Vocabulary file not found at {vocab_path}. Falling back to hashing.")
        return {}, {}
    
    def _encode_sentences(self, sentences_tokens):
        """Encode tokenized sentences using word2idx; fallback to hashing if vocab missing."""
        from nltk.tokenize import word_tokenize
        encoded = []
        use_hash = len(self.word2idx) == 0
        for sent in sentences_tokens:
            words = sent if isinstance(sent, list) else word_tokenize(str(sent).lower())
            if use_hash:
                encoded.append([hash(w) % 10000 for w in words])
            else:
                encoded.append([self.word2idx.get(w, self.word2idx.get('<UNK>', 1)) for w in words])
        return encoded
    
    def summarize_text(self, text: str, max_summary_length: int = 5, lang: str = 'english',
                        strategy: str = 'greedy', redundancy_penalty: float = 0.3,
                        dedup: bool = True, reorder_by_position: bool = True,
                        sbert_mmr: bool = False, mmr_alpha: float = 0.6) -> str:
        """Generate extractive summary for input text"""
        from nltk.tokenize import sent_tokenize
        
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(text, language=lang)
        except Exception:
            sentences = sent_tokenize(text)
        if not sentences:
            return ""
        
        # Tokenize and encode
        tokenized_sentences = [s.lower() for s in sentences]
        encoded_sentences = self._encode_sentences(tokenized_sentences)
        
        # Truncate doc length if needed
        max_doc_len = self.config.get('model.max_doc_len', 500)
        if len(encoded_sentences) > max_doc_len:
            encoded_sentences = encoded_sentences[:max_doc_len]
            sentences = sentences[:max_doc_len]
        
        # Convert to tensor
        max_sent_len = max(len(s) for s in encoded_sentences) if encoded_sentences else 1
        padded_sentences = [s + [0] * (max_sent_len - len(s)) for s in encoded_sentences]
        sentences_tensor = torch.tensor([padded_sentences], dtype=torch.long).to(self.device)
        mask = torch.ones(1, len(encoded_sentences), dtype=torch.float).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            selected_indices = self.model.extract_summary(
                sentences_tensor, mask, max_summary_length=max_summary_length,
                strategy=strategy, redundancy_penalty=redundancy_penalty
            )
        
        # Extract selected sentences (opcionalmente reordenadas por posición original)
        idxs = selected_indices[0].cpu().numpy().tolist()
        if reorder_by_position:
            idxs = sorted(set([i for i in idxs if 0 <= i < len(sentences)]))
        selected_sentences = [sentences[i] for i in idxs if 0 <= i < len(sentences)]
        
        # Simple deduplication: remove sentences that are near-duplicates of previous ones
        if dedup and selected_sentences:
            filtered = []
            seen = set()
            for s in selected_sentences:
                norm = ' '.join(s.lower().split())
                if norm not in seen:
                    seen.add(norm)
                    filtered.append(s)
            selected_sentences = filtered

        # Heurística opcional SBERT + MMR para reducir redundancia
        if sbert_mmr and selected_sentences:
            try:
                from src.fusion import sentence_embeddings, mmr_select
                embs = sentence_embeddings(selected_sentences)
                k = min(max_summary_length, len(selected_sentences))
                keep = mmr_select(embs, top_k=k, alpha=mmr_alpha)
                keep = sorted(keep)  # mantener orden relativo
                selected_sentences = [selected_sentences[i] for i in keep]
            except Exception as e:
                logger.warning(f"SBERT/MMR no disponible: {e}")
        
        return ' '.join(selected_sentences)

def evaluate_model(checkpoint_path: str, config_path: str = None, 
                  test_split: str = 'test', max_examples: int = None):
    """Evaluate model on test set"""
    config = Config(config_path)
    device = config.get('device.device', torch.device('cpu'))
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocabulary size from checkpoint or load data to determine
    try:
        vocab_size = checkpoint.get('vocab_size', None)
        if vocab_size is None:
            # Load a small sample to get vocabulary size
            temp_dataset = BookSumDataset(config, split='train', max_docs=10)
            vocab_size = len(temp_dataset.word2idx)
    except:
        vocab_size = 10000  # Default fallback
    
    model = MemSum(vocab_size, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    
    # Create data loader
    if test_split == 'test':
        _, _, test_loader, _ = create_data_loaders(config)
    elif test_split == 'validation':
        _, test_loader, _, _ = create_data_loaders(config)
    else:  # train
        test_loader, _, _, _ = create_data_loaders(config)
    
    # ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Try to import BERTScore (optional)
    try:
        from bert_score import score as bert_score_func
        use_bertscore = True
        logger.info("BERTScore disponible - será calculado")
    except ImportError:
        use_bertscore = False
        logger.warning("BERTScore no disponible. Instala con: pip install bert-score")
    
    # Evaluation
    all_rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    all_bert_scores = {'precision': [], 'recall': [], 'f1': []}
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if max_examples and i >= max_examples:
                break
                
            sentences = batch['sentences'].to(device)
            mask = batch['masks'].to(device)
            raw_sentences = batch['raw_sentences']
            raw_summaries = batch['raw_summaries']
            
            # Generate summaries
            predicted_indices = model.extract_summary(
                sentences, mask, 
                max_summary_length=config.get('data.max_summary_length', 10)
            )
            
            # Compute ROUGE scores
            batch_size = sentences.shape[0]
            for j in range(batch_size):
                pred_indices = predicted_indices[j].cpu().numpy()
                pred_summary = [raw_sentences[j][idx] for idx in pred_indices if idx < len(raw_sentences[j])]
                pred_text = ' '.join(pred_summary)
                target_text = ' '.join(raw_summaries[j])
                
                all_predictions.append(pred_text)
                all_targets.append(target_text)
                
                if pred_text.strip() and target_text.strip():
                    scores = rouge_scorer_obj.score(target_text, pred_text)
                    all_rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                    all_rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                    all_rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # Calculate average ROUGE scores
    avg_scores = {}
    for metric in all_rouge_scores:
        if all_rouge_scores[metric]:
            avg_scores[metric] = sum(all_rouge_scores[metric]) / len(all_rouge_scores[metric])
        else:
            avg_scores[metric] = 0.0
    
    # Calculate BERTScore if available
    if use_bertscore and all_predictions and all_targets:
        logger.info("Calculando BERTScore... (puede tardar varios minutos)")
        try:
            # Filter out empty predictions/targets
            valid_pairs = [(p, t) for p, t in zip(all_predictions, all_targets) 
                          if p.strip() and t.strip()]
            if valid_pairs:
                preds, refs = zip(*valid_pairs)
                P, R, F1 = bert_score_func(list(preds), list(refs), lang='en', verbose=False)
                avg_scores['bertscore_precision'] = P.mean().item()
                avg_scores['bertscore_recall'] = R.mean().item()
                avg_scores['bertscore_f1'] = F1.mean().item()
                logger.info(f"BERTScore F1: {avg_scores['bertscore_f1']:.4f}")
        except Exception as e:
            logger.error(f"Error calculando BERTScore: {e}")
            avg_scores['bertscore_precision'] = 0.0
            avg_scores['bertscore_recall'] = 0.0
            avg_scores['bertscore_f1'] = 0.0
    
    # Calculate SummaQA-like score (simplified content coverage)
    # Esta es una versión simplificada ya que SummaQA completo requiere modelos QA
    try:
        avg_scores['content_coverage'] = calculate_content_coverage(all_predictions, all_targets)
        logger.info(f"Content Coverage: {avg_scores['content_coverage']:.4f}")
    except Exception as e:
        logger.warning(f"No se pudo calcular Content Coverage: {e}")
        avg_scores['content_coverage'] = 0.0
    
    return avg_scores, all_predictions, all_targets

def calculate_content_coverage(predictions, targets):
    """
    Calcula una métrica simplificada de cobertura de contenido.
    Inspirada en SummaQA pero sin el componente completo de QA.
    
    Mide qué porcentaje de entidades/palabras clave del target
    aparecen en la predicción.
    """
    if not predictions or not targets:
        return 0.0
    
    total_coverage = 0.0
    count = 0
    
    for pred, target in zip(predictions, targets):
        if not pred.strip() or not target.strip():
            continue
        
        # Extract important words (simple heuristic: words longer than 4 chars)
        target_words = set(w.lower() for w in target.split() if len(w) > 4)
        pred_words = set(w.lower() for w in pred.split() if len(w) > 4)
        
        if target_words:
            coverage = len(target_words & pred_words) / len(target_words)
            total_coverage += coverage
            count += 1
    
    return total_coverage / count if count > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description='Evaluate MemSum model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'validation', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to evaluate')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to summarize (interactive mode)')
    
    args = parser.parse_args()
    
    if args.text:
        # Interactive summarization
        try:
            inference = MemSumInference(args.checkpoint, args.config)
            summary = inference.summarize_text(args.text)
            print("\nInput text:")
            print(args.text)
            print("\nGenerated summary:")
            print(summary)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
    else:
        # Evaluation mode
        try:
            logger.info(f"Evaluating model on {args.split} split...")
            avg_scores, predictions, targets = evaluate_model(
                args.checkpoint, args.config, args.split, args.max_examples
            )
            
            logger.info("Evaluation Results:")
            logger.info("=" * 60)
            logger.info("ROUGE Metrics:")
            logger.info(f"  ROUGE-1: {avg_scores.get('rouge1', 0.0):.4f}")
            logger.info(f"  ROUGE-2: {avg_scores.get('rouge2', 0.0):.4f}")
            logger.info(f"  ROUGE-L: {avg_scores.get('rougeL', 0.0):.4f}")
            
            if 'bertscore_f1' in avg_scores and avg_scores['bertscore_f1'] > 0:
                logger.info("\nBERTScore Metrics:")
                logger.info(f"  Precision: {avg_scores.get('bertscore_precision', 0.0):.4f}")
                logger.info(f"  Recall:    {avg_scores.get('bertscore_recall', 0.0):.4f}")
                logger.info(f"  F1:        {avg_scores.get('bertscore_f1', 0.0):.4f}")
            
            if 'content_coverage' in avg_scores and avg_scores['content_coverage'] > 0:
                logger.info("\nContent Coverage (SummaQA-like):")
                logger.info(f"  Coverage:  {avg_scores.get('content_coverage', 0.0):.4f}")
            
            logger.info("=" * 60)
            
            # Save results
            results = {
                'metrics': avg_scores,
                'num_examples': len(predictions),
                'checkpoint': args.checkpoint,
                'split': args.split,
                'summary': {
                    'rouge': {
                        'rouge1': avg_scores.get('rouge1', 0.0),
                        'rouge2': avg_scores.get('rouge2', 0.0),
                        'rougeL': avg_scores.get('rougeL', 0.0)
                    },
                    'bertscore': {
                        'precision': avg_scores.get('bertscore_precision', 0.0),
                        'recall': avg_scores.get('bertscore_recall', 0.0),
                        'f1': avg_scores.get('bertscore_f1', 0.0)
                    } if 'bertscore_f1' in avg_scores else None,
                    'content_coverage': avg_scores.get('content_coverage', 0.0)
                }
            }
            
            if args.save_predictions:
                results['predictions'] = predictions[:100]  # Save first 100 predictions
                results['targets'] = targets[:100]
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {args.output}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

if __name__ == '__main__':
    main()