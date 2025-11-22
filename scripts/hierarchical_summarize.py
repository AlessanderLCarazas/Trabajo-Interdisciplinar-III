#!/usr/bin/env python3
"""
Resumen jerárquico de textos largos o PDFs con MemSum.

Pipeline:
1) Particionar el documento en chunks de N oraciones (p.ej., 500)
2) Resumir cada chunk con MemSum (pasada 1)
3) Fusionar resúmenes parciales con una segunda pasada de MemSum

Este script NO añade dependencias nuevas: reutiliza MemSumInference.

Uso:
  python scripts/hierarchical_summarize.py --input path/al/texto.txt \
      --checkpoint checkpoints/best_model.pt \
      --config configs/booksum_config.yaml \
      --chunk_size 500 --overlap 20 --lang spanish \
      --chunk_summary_len 6 --final_summary_len 8

  # O directo desde PDF
  python scripts/hierarchical_summarize.py --input pruebas/Mother\ Tongue\ by\ Tan.pdf \
      --is_pdf --final_summary_len 8

Salida: imprime por pantalla el resumen final y, opcionalmente, guarda archivos.
"""

import os
import sys
import argparse
from typing import List

# Asegurar rutas
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'src'))

from evaluate import MemSumInference  # type: ignore


def extract_text_pdf(pdf_path: str) -> str:
    """Extraer texto de PDF con pdfminer.six; fallback a PyPDF2 si falla."""
    text = ''
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(pdf_path) or ''
    except Exception:
        pass
    if not text.strip():
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                chunks = []
                for page in reader.pages:
                    try:
                        t = page.extract_text() or ''
                    except Exception:
                        t = ''
                    if t:
                        chunks.append(' '.join(t.split()))
                text = '\n'.join(chunks)
        except Exception:
            text = ''
    return text


def sent_tokenize_lang(text: str, lang: str = 'spanish') -> List[str]:
    from nltk.tokenize import sent_tokenize
    try:
        return sent_tokenize(text, language=lang)
    except Exception:
        return sent_tokenize(text)


def chunk_sentences(sents: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    """Particionar en ventanas deslizantes de chunk_size con solape overlap."""
    if not sents:
        return []
    chunks = []
    i = 0
    n = len(sents)
    step = max(1, chunk_size - overlap)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(sents[i:j])
        if j == n:
            break
        i += step
    return chunks


def main():
    p = argparse.ArgumentParser(description='Resumen jerárquico con MemSum')
    p.add_argument('--input', type=str, required=True, help='Ruta a .txt o .pdf')
    p.add_argument('--is_pdf', action='store_true', help='Forzar lectura como PDF')
    p.add_argument('--checkpoint', type=str, default=os.path.join(ROOT, 'checkpoints', 'best_model.pt'))
    p.add_argument('--config', type=str, default=os.path.join(ROOT, 'configs', 'booksum_config.yaml'))
    p.add_argument('--lang', type=str, default='spanish', help='Idioma para segmentación de oraciones')
    p.add_argument('--chunk_size', type=int, default=500, help='Máximo de oraciones por chunk')
    p.add_argument('--overlap', type=int, default=20, help='Oraciones de solape entre chunks')
    p.add_argument('--chunk_summary_len', type=int, default=6, help='Oraciones por resumen de cada chunk')
    p.add_argument('--final_summary_len', type=int, default=8, help='Oraciones del resumen final (segunda pasada)')
    p.add_argument('--strategy', type=str, default='greedy', choices=['greedy', 'sample'], help='Estrategia de selección')
    p.add_argument('--redundancy_penalty', type=float, default=0.3, help='Penalización de redundancia en cada pasada')
    p.add_argument('--sbert_mmr', action='store_true', help='Aplicar deduplicación SBERT+MMR en los resúmenes (requiere sentence-transformers)')
    p.add_argument('--mmr_alpha', type=float, default=0.6, help='Alpha para MMR (0..1): relevancia vs novedad')
    p.add_argument('--save_partials', action='store_true', help='Guardar resúmenes parciales junto al final')
    p.add_argument('--output', type=str, default=None, help='Archivo donde guardar el resumen final')
    args = p.parse_args()

    # Leer documento
    path = args.input
    if not os.path.exists(path):
        print(f"[ERROR] No existe: {path}")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    is_pdf = args.is_pdf or ext == '.pdf'

    if is_pdf:
        text = extract_text_pdf(path)
        if not text.strip():
            print('[ERROR] No se pudo extraer texto del PDF')
            sys.exit(2)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

    # Segmentar a oraciones y particionar
    sentences = sent_tokenize_lang(text, lang=args.lang)
    if not sentences:
        print('[ERROR] El texto está vacío o no se pudo segmentar.')
        sys.exit(3)

    chunks = chunk_sentences(sentences, args.chunk_size, args.overlap)
    print(f"[INFO] Oraciones totales: {len(sentences)} | Chunks: {len(chunks)}")

    # Crear inferencia
    infer = MemSumInference(args.checkpoint, args.config)

    # Pasada 1: resumen por chunk
    partial_summaries: List[str] = []
    for k, chunk in enumerate(chunks, 1):
        chunk_text = ' '.join(chunk)
        summary_k = infer.summarize_text(
            chunk_text,
            max_summary_length=args.chunk_summary_len,
            lang=args.lang,
            strategy=args.strategy,
            redundancy_penalty=args.redundancy_penalty,
            dedup=True,
            reorder_by_position=True,
            sbert_mmr=args.sbert_mmr,
            mmr_alpha=args.mmr_alpha,
        )
        partial_summaries.append(summary_k)
        print(f"[Chunk {k}/{len(chunks)}] {len(chunk)} oraciones → resumen de {len(summary_k.split('.'))} sents aprox.")

    # Guardar parciales opcionalmente
    if args.save_partials:
        base = os.path.splitext(args.output or path)[0]
        os.makedirs(os.path.dirname(base), exist_ok=True) if os.path.dirname(base) else None
        for i, summ in enumerate(partial_summaries, 1):
            out_i = f"{base}.partial_{i}.txt"
            with open(out_i, 'w', encoding='utf-8') as f:
                f.write(summ)
        print(f"[OK] Resúmenes parciales guardados con prefijo: {base}.partial_*.txt")

    # Pasada 2: fusión jerárquica (resumir los resúmenes)
    fusion_text = ' '.join(partial_summaries)
    final_summary = infer.summarize_text(
        fusion_text,
        max_summary_length=args.final_summary_len,
        lang=args.lang,
        strategy=args.strategy,
        redundancy_penalty=args.redundancy_penalty,
        dedup=True,
        reorder_by_position=True,
        sbert_mmr=args.sbert_mmr,
        mmr_alpha=args.mmr_alpha,
    )

    print("\n===== RESUMEN FINAL =====\n")
    print(final_summary)

    if args.output:
        out_dir = os.path.dirname(args.output)
        os.makedirs(out_dir, exist_ok=True) if out_dir else None
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(final_summary)
        print(f"\n[OK] Resumen final guardado en: {args.output}")


if __name__ == '__main__':
    main()
