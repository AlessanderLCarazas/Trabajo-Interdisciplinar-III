"""
ENTRENADOR CON REINFORCEMENT LEARNING - Motor del aprendizaje
==============================================================
Este archivo maneja todo el proceso de entrenamiento del modelo MemSum.

QUÉ HACE:
- Entrena el modelo usando REINFORCE (algoritmo de RL)
- Calcula recompensas basadas en ROUGE-L y solapamiento con resúmenes oracle
- Maneja mixed precision, gradient accumulation y clipping
- Evalúa el modelo en validation set cada cierto número de pasos
- Guarda checkpoints y el mejor modelo automáticamente

COMPONENTES CLAVE:
- compute_rouge_reward(): Calcula recompensa comparando con resumen gold
- compute_oracle_reward(): Recompensa por seleccionar oraciones oracle
- train_step(): Un paso de entrenamiento (forward + backward + update)
- evaluate(): Evaluación completa en validation/test set
- save_checkpoint(): Guarda estado completo del modelo y optimizador

ALGORITMO:
- Usa policy gradients (REINFORCE) para aprender qué oraciones extraer
- Combina recompensas: 70% ROUGE-L + 30% oracle overlap
- Optimizador Adam con learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import logging
from tqdm import tqdm
import json
from rouge_score import rouge_scorer
from typing import Dict, List, Tuple, Optional
import wandb

logger = logging.getLogger(__name__)

class RLTrainer:
    """Reinforcement Learning trainer for MemSum"""
    
    def __init__(self, model, config, vocab_size):
        self.model = model
        self.config = config
        self.device = config.get('device.device', torch.device('cpu'))
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('training.learning_rate', 1e-4),
            weight_decay=1e-6
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Mixed precision training
        self.use_amp = config.get('device.mixed_precision', True)
        if self.use_amp:
            try:
                # Try new API first
                from torch.amp import GradScaler as NewGradScaler
                self.scaler = NewGradScaler('cuda')
            except (ImportError, TypeError):
                # Fallback to old API
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Training parameters
        self.gamma = config.get('rl.gamma', 0.95)
        self.entropy_coef = config.get('rl.entropy_coef', 0.01)
        self.value_coef = config.get('rl.value_coef', 0.5)
        self.gradient_clip_norm = config.get('training.gradient_clip_norm', 5.0)
        self.accumulation_steps = config.get('training.accumulation_steps', 4)
        # Supervised warmup (imitation learning) to stabilize early training
        self.sup_warmup_epochs = config.get('training.supervised_warmup_epochs', 0)
        self.sup_lambda = config.get('training.supervised_lambda', 0.0)
        
        # ROUGE scorer for evaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        
        # Logging
        self.use_wandb = config.get('logging.use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config.get('logging.wandb_project', 'memsum-booksum'),
                config=config.config
            )
        
        # Checkpoint directory
        self.checkpoint_dir = config.get('paths.checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_rouge = 0.0
        self.patience_counter = 0
        self.current_epoch = 0  # will be set in train()
        
    def compute_rouge_reward(self, predicted_indices: torch.Tensor, 
                           oracle_indices: torch.Tensor,
                           raw_sentences: List[List[str]], 
                           raw_summaries: List[List[str]]) -> torch.Tensor:
        """Compute ROUGE-based reward for predicted summaries"""
        batch_size = predicted_indices.shape[0]
        rewards = []
        
        for i in range(batch_size):
            # Get predicted summary
            pred_indices = predicted_indices[i].cpu().numpy()
            pred_summary = [raw_sentences[i][idx] for idx in pred_indices if idx < len(raw_sentences[i])]
            pred_text = ' '.join(pred_summary)
            
            # Get target summary
            target_text = ' '.join(raw_summaries[i])
            
            # Compute ROUGE score
            try:
                if pred_text.strip() and target_text.strip():
                    scores = self.rouge_scorer.score(target_text, pred_text)
                    rouge_l = scores['rougeL'].fmeasure
                else:
                    rouge_l = 0.1  # Small baseline reward
            except:
                rouge_l = 0.1  # Fallback reward
            
            rewards.append(rouge_l)
        
        return torch.tensor(rewards, dtype=torch.float, device=self.device)
    
    def compute_oracle_reward(self, predicted_indices: torch.Tensor,
                            oracle_indices: torch.Tensor) -> torch.Tensor:
        """Compute reward based on overlap with oracle summary"""
        batch_size = predicted_indices.shape[0]
        rewards = []
        
        for i in range(batch_size):
            pred_set = set(predicted_indices[i].cpu().numpy())
            oracle_set = set(oracle_indices[i].cpu().numpy())
            oracle_set = {idx for idx in oracle_set if idx >= 0}  # Remove padding
            
            if len(oracle_set) > 0:
                overlap = len(pred_set.intersection(oracle_set))
                precision = overlap / len(pred_set) if len(pred_set) > 0 else 0
                recall = overlap / len(oracle_set)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                reward = f1
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float, device=self.device)
    
    def compute_policy_loss(self, action_logits: torch.Tensor, 
                          selected_actions: torch.Tensor,
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute policy gradient loss with advantage estimation"""
        batch_size = selected_actions.shape[0]
        max_steps = selected_actions.shape[1] if len(selected_actions.shape) > 1 else 1
        
        # Ensure selected_actions is 2D
        if len(selected_actions.shape) == 1:
            selected_actions = selected_actions.unsqueeze(1)
        
        # Compute advantages - use value head as baseline
        predicted_values = values.mean(dim=1) if len(values.shape) > 1 else values  # [batch]
        advantages = rewards - predicted_values.detach()
        
        # Compute action log probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Gather log probabilities for selected actions
        selected_log_probs_list = []
        
        for step in range(max_steps):
            if step < selected_actions.shape[1]:
                batch_indices = torch.arange(batch_size, device=self.device)
                action_indices = selected_actions[:, step]
                
                # Filter out invalid indices
                valid_mask = (action_indices >= 0) & (action_indices < action_log_probs.shape[1])
                
                step_log_probs = torch.zeros(batch_size, device=self.device)
                if valid_mask.any():
                    valid_batch_indices = batch_indices[valid_mask]
                    valid_action_indices = action_indices[valid_mask]
                    step_log_probs[valid_mask] = action_log_probs[
                        valid_batch_indices, valid_action_indices
                    ]
                selected_log_probs_list.append(step_log_probs)
        
        if selected_log_probs_list:
            # Average over steps
            selected_log_probs = torch.stack(selected_log_probs_list, dim=1).mean(dim=1)  # [batch_size]
        else:
            selected_log_probs = torch.zeros(batch_size, device=self.device)
        
        # Policy loss (negative because we want to maximize reward)
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        
        # Value loss - compare mean values with rewards
        value_loss = F.mse_loss(predicted_values, rewards)
        
        # Entropy loss for exploration
        entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()
        entropy_loss = -entropy
        
        return policy_loss, value_loss, entropy_loss

    def compute_supervised_loss(self, action_logits: torch.Tensor, oracle_indices: torch.Tensor) -> torch.Tensor:
        """KLDiv between model policy and oracle uniform distribution over oracle indices.
        oracle_indices: [B, L] with -1 padding.
        """
        B, L = action_logits.shape
        # Build target distribution
        target = torch.zeros(B, L, device=action_logits.device, dtype=torch.float)
        valid_batch = torch.zeros(B, dtype=torch.bool, device=action_logits.device)
        for i in range(B):
            idxs = oracle_indices[i]
            idxs = idxs[(idxs >= 0) & (idxs < L)]
            if idxs.numel() > 0:
                target[i, idxs] = 1.0 / idxs.numel()
                valid_batch[i] = True
        if not valid_batch.any():
            return torch.tensor(0.0, device=action_logits.device)
        log_probs = F.log_softmax(action_logits, dim=-1)
        # KLDivLoss expects input log-prob and target prob
        kld = F.kl_div(log_probs[valid_batch], target[valid_batch], reduction='batchmean')
        return kld
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        sentences = batch['sentences'].to(self.device)
        oracle_indices = batch['oracle_indices'].to(self.device)
        mask = batch['masks'].to(self.device)
        raw_sentences = batch['raw_sentences']
        raw_summaries = batch['raw_summaries']
        
        batch_size = sentences.shape[0]
        max_summary_length = self.config.get('data.max_summary_length', 10)
        
        try:
            # Try new API first
            from torch.amp import autocast as new_autocast
            autocast_context = new_autocast('cuda', enabled=self.use_amp)
        except (ImportError, TypeError):
            # Fallback to old API
            from torch.cuda.amp import autocast
            autocast_context = autocast(enabled=self.use_amp)
        
        with autocast_context:
            # Extract summary using current policy (no_grad to save memory)
            with torch.no_grad():
                predicted_indices = self.model.extract_summary(
                    sentences, mask, max_summary_length=max_summary_length
                )
            
            # Get model outputs for the extraction process
            extraction_history = torch.zeros_like(mask)
            memory_state = None
            all_action_logits = []
            all_values = []
            
            # Get action logits and values from a single forward pass
            outputs = self.model(sentences, mask, extraction_history, memory_state)
            action_logits = outputs['action_logits']  # [batch_size, max_doc_len]
            values = outputs['values']  # [batch_size, max_doc_len]
            
            # Compute rewards
            rouge_rewards = self.compute_rouge_reward(
                predicted_indices, oracle_indices, raw_sentences, raw_summaries
            )
            oracle_rewards = self.compute_oracle_reward(predicted_indices, oracle_indices)
            
            # Combine rewards
            rewards = 0.7 * rouge_rewards + 0.3 * oracle_rewards
            
            # Compute losses
            policy_loss, value_loss, entropy_loss = self.compute_policy_loss(
                action_logits, predicted_indices, rewards, values, mask
            )
            # Optional supervised warmup loss to guide policy early on
            sup_loss = torch.tensor(0.0, device=self.device)
            if self.sup_warmup_epochs and self.current_epoch <= self.sup_warmup_epochs and self.sup_lambda > 0:
                sup_loss = self.compute_supervised_loss(action_logits, oracle_indices)
            
        # Total loss with NaN protection
        policy_loss = torch.nan_to_num(policy_loss, nan=0.0)
        value_loss = torch.nan_to_num(value_loss, nan=0.0)
        entropy_loss = torch.nan_to_num(entropy_loss, nan=0.0)
        sup_loss = torch.nan_to_num(sup_loss, nan=0.0)
        
        total_loss = (
            policy_loss + 
            self.value_coef * value_loss + 
            self.entropy_coef * entropy_loss +
            self.sup_lambda * sup_loss
        )        # Backward pass with gradient scaling
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            
            if (self.global_step + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss.backward()
            
            if (self.global_step + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_rouge_reward': rouge_rewards.mean().item(),
            'avg_oracle_reward': oracle_rewards.mean().item(),
            'sup_loss': sup_loss.item() if isinstance(sup_loss, torch.Tensor) else float(sup_loss)
        }
    
    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate model on validation set
        Uses config keys:
          - evaluation.use_bertscore: bool (default False)
          - evaluation.max_val_samples: int or None (limit number of items evaluated)
        """
        self.model.eval()
        total_rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        total_samples = 0
        
        # Optional: BERTScore (config gated to avoid heavy model loading)
        use_bertscore = bool(self.config.get('evaluation.use_bertscore', False))
        bertscore = None
        if use_bertscore:
            try:
                from bert_score import score as bertscore_fn
            except Exception:
                use_bertscore = False

        # Optional: limit number of validation samples to speed up evaluation
        max_val_samples = self.config.get('evaluation.max_val_samples', None)
        seen_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                sentences = batch['sentences'].to(self.device)
                mask = batch['masks'].to(self.device)
                raw_sentences = batch['raw_sentences']
                raw_summaries = batch['raw_summaries']
                
                # Extract summaries
                predicted_indices = self.model.extract_summary(
                    sentences, mask, 
                    max_summary_length=self.config.get('data.max_summary_length', 10)
                )
                
                # Compute ROUGE scores
                batch_size = sentences.shape[0]
                preds_for_bert = []
                refs_for_bert = []
                for i in range(batch_size):
                    pred_indices = predicted_indices[i].cpu().numpy()
                    pred_summary = [raw_sentences[i][idx] for idx in pred_indices if idx < len(raw_sentences[i])]
                    pred_text = ' '.join(pred_summary)
                    target_text = ' '.join(raw_summaries[i])
                    
                    if pred_text.strip() and target_text.strip():
                        scores = self.rouge_scorer.score(target_text, pred_text)
                        total_rouge_scores['rouge1'] += scores['rouge1'].fmeasure
                        total_rouge_scores['rouge2'] += scores['rouge2'].fmeasure
                        total_rouge_scores['rougeL'] += scores['rougeL'].fmeasure
                        total_samples += 1
                        if use_bertscore:
                            preds_for_bert.append(pred_text)
                            refs_for_bert.append(target_text)

                # Respect evaluation sample budget - check BEFORE processing to avoid overshoot
                seen_samples += batch_size
                should_break = max_val_samples is not None and isinstance(max_val_samples, int) and seen_samples >= max_val_samples
                
                # Batch-level BERTScore to avoid huge memory (only if not breaking)
                if use_bertscore and preds_for_bert and not should_break:
                    try:
                        P, R, F = bertscore_fn(preds_for_bert, refs_for_bert, lang='en', verbose=False)
                        if bertscore is None:
                            bertscore = []
                        bertscore.extend(F.tolist())
                    except Exception:
                        pass
                
                # Break after processing current batch if budget exceeded
                if should_break:
                    # Final BERTScore computation if needed
                    if use_bertscore and preds_for_bert:
                        try:
                            P, R, F = bertscore_fn(preds_for_bert, refs_for_bert, lang='en', verbose=False)
                            if bertscore is None:
                                bertscore = []
                            bertscore.extend(F.tolist())
                        except Exception:
                            pass
                    break
        
        # Average scores
        if total_samples > 0:
            avg_scores = {k: v / total_samples for k, v in total_rouge_scores.items()}
        else:
            avg_scores = {k: 0.0 for k in total_rouge_scores.keys()}
        
        if use_bertscore and bertscore:
            avg_scores['bertscore_f1'] = float(np.mean(bertscore))
        self.model.train()
        return avg_scores
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.config,
            'metrics': metrics,
            'best_rouge': self.best_rouge
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with ROUGE-L: {metrics.get('rougeL', 0):.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.best_rouge = checkpoint.get('best_rouge', 0.0)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0)
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Main training loop"""
        logger.info("Starting training...")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_losses = []
            epoch_rewards = []
            
            # Training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in train_pbar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['total_loss'])
                epoch_rewards.append(metrics['avg_reward'])
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'reward': f"{metrics['avg_reward']:.4f}",
                    'rouge': f"{metrics['avg_rouge_reward']:.4f}"
                })
                
                # Logging
                if self.use_wandb:
                    wandb.log(metrics, step=self.global_step)
                
                self.global_step += 1
                
                # Evaluation during training
                eval_every = self.config.get('training.eval_every', 1000)
                if isinstance(eval_every, int) and eval_every > 0 and self.global_step % eval_every == 0:
                    val_metrics = self.evaluate(val_loader)
                    logger.info(f"Step {self.global_step} - Validation ROUGE: {val_metrics}")
                    
                    if self.use_wandb:
                        wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, step=self.global_step)
            
            # End of epoch evaluation
            val_metrics = self.evaluate(val_loader)
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)
            
            logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
            logger.info(f"Epoch {epoch+1} - Validation ROUGE: {val_metrics}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics.get('rougeL', 0))
            
            # Save checkpoint
            is_best = val_metrics.get('rougeL', 0) > self.best_rouge
            if is_best:
                self.best_rouge = val_metrics.get('rougeL', 0)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config.get('training.save_every', 5) == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.get('training.early_stopping_patience', 5):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'train_reward': avg_reward,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
        
        logger.info("Training completed!")
        return self.best_rouge