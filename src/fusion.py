"""
Fusion heurística y deduplicación con SBERT + MMR.
- sentence_embeddings(): calcula embeddings de oraciones con Sentence-BERT
- mmr_select(): selección con Maximal Marginal Relevance

Uso típico:
  embs = sentence_embeddings(sentences)
  selected_ids = mmr_select(embs, top_k=k, alpha=0.6)
"""
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
except Exception:  # pragma: no cover
    SentenceTransformer = None
    torch = None


def sentence_embeddings(sentences: List[str], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: Optional[str] = None) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers no está instalado. Añádelo a requirements.txt y pip install.")
    model = SentenceTransformer(model_name)
    if device and hasattr(model, 'to'):
        try:
            model.to(device)
        except Exception:
            pass
    embs = model.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32)


def mmr_select(embeddings: np.ndarray, top_k: int, alpha: float = 0.6) -> List[int]:
    """Maximal Marginal Relevance.
    - embeddings: (N, D) normalizados
    - alpha: trade-off relevancia (a centroid) vs novedad (1-alpha)
    """
    N = embeddings.shape[0]
    if N == 0:
        return []
    top_k = max(1, min(top_k, N))

    # Centroid relevance
    centroid = embeddings.mean(axis=0, keepdims=True)
    centroid = centroid / max(1e-8, np.linalg.norm(centroid))
    rel = (embeddings @ centroid.T).squeeze(-1)  # (N,)

    selected: List[int] = []
    candidates = list(range(N))

    # Precompute similarity among sentences
    sim = embeddings @ embeddings.T  # (N, N)

    for _ in range(top_k):
        if not candidates:
            break
        if not selected:
            # pick most relevant first
            i = int(np.argmax(rel[candidates]))
            chosen = candidates[i]
            selected.append(chosen)
            candidates.remove(chosen)
            continue
        # novelty: min similarity to already selected
        max_sim_to_selected = np.max(sim[candidates][:, selected], axis=1)
        # MMR score
        mmr_scores = alpha * rel[candidates] - (1 - alpha) * max_sim_to_selected
        i = int(np.argmax(mmr_scores))
        chosen = candidates[i]
        selected.append(chosen)
        candidates.remove(chosen)

    return selected
