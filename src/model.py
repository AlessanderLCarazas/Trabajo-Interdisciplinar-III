"""
ARQUITECTURA DEL MODELO MEMSUM - Núcleo de la red neuronal
============================================================
Este archivo define toda la arquitectura del modelo MemSum para resumir documentos.

QUÉ CONTIENE:
- SentenceEncoder: Codifica cada oración con BiLSTM + LayerNorm
- DocumentEncoder: Codifica el documento completo con Transformer
- MemoryModule: Módulo de memoria con GRU bidireccional + atención
- ExtractionPolicy: Política para decidir qué oraciones extraer (policy + value)
- MemSum: Clase principal que combina todos los componentes

CÓMO FUNCIONA:
1. Cada oración se codifica con BiLSTM
2. El documento se procesa con Transformer (4 capas)
3. El módulo de memoria mantiene estado entre decisiones
4. La política decide qué oraciones incluir en el resumen

TÉCNICAS USADAS:
- Reinforcement Learning (REINFORCE)
- Attention mechanisms
- Memory-augmented networks
- Extractive summarization (selecciona oraciones existentes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional
import os
import pickle

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, word_embed_dim: int, sent_embed_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            word_embed_dim, 
            sent_embed_dim // 2, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(sent_embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sentences: torch.Tensor, sentence_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_doc_len, max_sent_len = sentences.shape
        
        sentences_flat = sentences.view(-1, max_sent_len)
        word_embeds = self.word_embedding(sentences_flat)
        word_embeds = self.dropout(word_embeds)
        
        lstm_out, (hidden, _) = self.lstm(word_embeds)
        sentence_repr = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        sentence_repr = self.layer_norm(sentence_repr)
        
        sentence_embeddings = sentence_repr.view(batch_size, max_doc_len, -1)
        return sentence_embeddings

class DocumentEncoder(nn.Module):
    def __init__(self, sent_embed_dim: int, doc_embed_dim: int, num_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.projection = nn.Linear(sent_embed_dim, doc_embed_dim)
        self.pos_encoding = PositionalEncoding(doc_embed_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=doc_embed_dim,
            nhead=num_heads,
            dim_feedforward=doc_embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(doc_embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sentence_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        doc_embeds = self.projection(sentence_embeddings)
        doc_embeds = self.dropout(doc_embeds)
        
        doc_embeds = doc_embeds.transpose(0, 1)
        doc_embeds = self.pos_encoding(doc_embeds)
        doc_embeds = doc_embeds.transpose(0, 1)
        
        attn_mask = (mask == 0)
        doc_embeddings = self.transformer(doc_embeds, src_key_padding_mask=attn_mask)
        doc_embeddings = self.final_norm(doc_embeddings)
        
        return doc_embeddings

class MemoryModule(nn.Module):
    def __init__(self, doc_embed_dim: int, memory_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.memory_update = nn.GRU(
            doc_embed_dim, 
            memory_dim, 
            batch_first=True,
            bidirectional=True
        )
        
        self.memory_projection = nn.Linear(memory_dim * 2, memory_dim)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(memory_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, doc_embeddings: torch.Tensor, extraction_history: torch.Tensor,
                memory_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        extracted_mask = extraction_history.unsqueeze(-1)
        extracted_embeddings = doc_embeddings * extracted_mask
        
        memory_output, new_memory_state = self.memory_update(extracted_embeddings, memory_state)
        memory_output = self.memory_projection(memory_output)
        
        attended_memory, _ = self.memory_attention(memory_output, memory_output, memory_output)
        
        gate_input = torch.cat([memory_output, attended_memory], dim=-1)
        gate_weights = self.gate(gate_input)
        
        memory_embeddings = gate_weights * memory_output + (1 - gate_weights) * attended_memory
        memory_embeddings = self.norm(memory_embeddings)
        memory_embeddings = self.dropout(memory_embeddings)
        
        return memory_embeddings, new_memory_state

class ExtractionPolicy(nn.Module):
    def __init__(self, doc_embed_dim: int, memory_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        input_dim = doc_embed_dim + memory_dim + 64
        self.position_embedding = nn.Embedding(1000, 64)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, doc_embeddings: torch.Tensor, memory_embeddings: torch.Tensor,
                positions: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        pos_embeds = self.position_embedding(positions)
        combined_features = torch.cat([doc_embeddings, memory_embeddings, pos_embeds], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        
        action_logits = self.policy_head(fused_features).squeeze(-1)
        values = self.value_head(fused_features).squeeze(-1)
        
        action_logits = action_logits.masked_fill(mask == 0, float('-inf'))
        values = values.masked_fill(mask == 0, 0)
        
        return action_logits, values

class MemSum(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        
        self.config = config
        word_embed_dim = config.get('model.word_embed_dim', 200)
        sent_embed_dim = config.get('model.sent_embed_dim', 256)
        doc_embed_dim = config.get('model.doc_embed_dim', 256)
        memory_dim = config.get('model.memory_dim', 256)
        hidden_dim = config.get('model.hidden_dim', 256)
        num_layers = config.get('model.num_layers', 2)
        dropout = config.get('model.dropout', 0.1)
        
        self.sentence_encoder = SentenceEncoder(
            vocab_size, word_embed_dim, sent_embed_dim, num_layers, dropout
        )
        
        self.document_encoder = DocumentEncoder(
            sent_embed_dim, doc_embed_dim, num_heads=8, num_layers=4, dropout=dropout
        )
        
        self.memory_module = MemoryModule(
            doc_embed_dim, memory_dim, dropout
        )
        
        self.extraction_policy = ExtractionPolicy(
            doc_embed_dim, memory_dim, hidden_dim, dropout
        )
        
        self.apply(self._init_weights)

        try:
            self._maybe_init_pretrained_embeddings(vocab_size)
        except Exception:
            pass
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)

    def _maybe_init_pretrained_embeddings(self, vocab_size: int):
        """Si hay glove_path en config y existe vocab.pkl, inicializa embeddings.
        Espera que la dimensión coincida con model.word_embed_dim.
        """
        glove_path = self.config.get('paths.glove_path', None)
        if not glove_path or not os.path.isfile(glove_path):
            return
        data_dir = self.config.get('paths.data_dir', './data')
        vocab_pkl = os.path.join(data_dir, 'vocab.pkl')
        if not os.path.isfile(vocab_pkl):
            return
        embed_dim = self.config.get('model.word_embed_dim', 200)

        # Cargar vocab
        with open(vocab_pkl, 'rb') as f:
            vocab = pickle.load(f)
        word2idx = vocab.get('word2idx', {})
        if not word2idx or len(word2idx) != vocab_size:
            # Evitar inconsistencias (por ejemplo, dummy vocab)
            return

        # Cargar GloVe
        matrix = np.random.normal(0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)
        found = 0
        with open(glove_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) < embed_dim + 1:
                    continue
                word = parts[0]
                vec = parts[1:1+embed_dim]
                try:
                    vec = np.asarray(vec, dtype=np.float32)
                except Exception:
                    continue
                idx = word2idx.get(word)
                if idx is not None and 0 <= idx < vocab_size:
                    matrix[idx] = vec
                    found += 1
        with torch.no_grad():
            w = torch.from_numpy(matrix)
            if self.sentence_encoder.word_embedding.weight.shape == w.shape:
                self.sentence_encoder.word_embedding.weight.copy_(w)
                if self.config.get('model.freeze_embeddings', False):
                    self.sentence_encoder.word_embedding.weight.requires_grad = False
        # Mensaje informativo (no usar logger aquí para simplicidad)
        try:
            print(f"[MemSum] GloVe inicializado: {found} palabras mapeadas de {vocab_size}.")
        except Exception:
            pass
            
    def forward(self, sentences: torch.Tensor, mask: torch.Tensor, 
                extraction_history: Optional[torch.Tensor] = None,
                memory_state: Optional[torch.Tensor] = None) -> dict:
        
        batch_size, max_doc_len, max_sent_len = sentences.shape
        device = sentences.device
        
        if extraction_history is None:
            extraction_history = torch.zeros(batch_size, max_doc_len, device=device)
        
        sentence_lengths = (sentences != 0).sum(dim=-1)
        sentence_embeddings = self.sentence_encoder(sentences, sentence_lengths)
        doc_embeddings = self.document_encoder(sentence_embeddings, mask)
        memory_embeddings, new_memory_state = self.memory_module(
            doc_embeddings, extraction_history, memory_state
        )
        
        positions = torch.arange(max_doc_len, device=device).unsqueeze(0).expand(batch_size, -1)
        action_logits, values = self.extraction_policy(
            doc_embeddings, memory_embeddings, positions, mask
        )
        
        return {
            'action_logits': action_logits,
            'values': values,
            'memory_embeddings': memory_embeddings,
            'new_memory_state': new_memory_state,
            'doc_embeddings': doc_embeddings
        }
    
    def extract_summary(self, sentences: torch.Tensor, mask: torch.Tensor, 
                       max_summary_length: int = 5, temperature: float = 1.0,
                       strategy: str = 'sample', redundancy_penalty: float = 0.0) -> torch.Tensor:
        
        batch_size, max_doc_len = sentences.shape[:2]
        device = sentences.device
        
        extraction_history = torch.zeros(batch_size, max_doc_len, device=device)
        memory_state = None
        selected_indices = []
        
        selected_embed_cache = None  # will keep mean embedding of selected sentences per batch

        for step in range(max_summary_length):
            outputs = self.forward(sentences, mask, extraction_history, memory_state)
            
            action_logits = outputs['action_logits'] / temperature
            memory_state = outputs['new_memory_state']
            
            # Optional redundancy penalty based on cosine similarity to already selected sentences
            if redundancy_penalty > 0 and selected_indices:
                with torch.no_grad():
                    doc_emb = outputs['doc_embeddings']  # [B, L, D]
                    # Build mask for valid candidates
                    valid_mask = (1 - extraction_history) * mask  # [B, L]
                    # Compute mean embedding of selected sentences per batch
                    B, L, D = doc_emb.shape
                    mean_sel = torch.zeros(B, D, device=doc_emb.device)
                    for b in range(B):
                        sel_idxs = [idx[b].item() for idx in selected_indices]
                        sel_idxs = [i for i in sel_idxs if 0 <= i < L]
                        if sel_idxs:
                            sel_emb = doc_emb[b, sel_idxs, :]
                            mean_sel[b] = F.normalize(sel_emb.mean(dim=0), dim=0)
                        else:
                            mean_sel[b] = 0.0
                    # Cosine similarity between each candidate and mean selected
                    cand_emb = F.normalize(doc_emb, dim=-1)  # [B, L, D]
                    sim = (cand_emb * mean_sel.unsqueeze(1)).sum(dim=-1)  # [B, L]
                    # Decrease logits where similarity is high
                    action_logits = action_logits - redundancy_penalty * sim

            action_probs = F.softmax(action_logits, dim=-1)
            action_probs = action_probs * (1 - extraction_history) * mask
            
            if action_probs.sum(dim=-1).max() == 0:
                break
                
            if strategy == 'greedy':
                selected_idx = torch.argmax(action_probs, dim=-1)
            else:
                selected_idx = torch.multinomial(action_probs + 1e-8, 1).squeeze(-1)
            selected_indices.append(selected_idx)
            
            batch_idx = torch.arange(batch_size, device=device)
            extraction_history[batch_idx, selected_idx] = 1
        
        if not selected_indices:
            return torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            
        return torch.stack(selected_indices, dim=1)
