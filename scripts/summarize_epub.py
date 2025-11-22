#!/usr/bin/env python3
"""
Resumir un archivo ePub con MemSum usando ebooklib + BeautifulSoup para extraer texto.

Uso:
  python scripts/summarize_epub.py libro.epub \
      --checkpoint checkpoints/best_model.pt \
      --config configs/booksum_config.yaml \
      --lang spanish --max_summary_len 8

Requiere: ebooklib, beautifulsoup4
"""
import os
import sys
import argparse

# Asegurar src en el path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'src'))

from evaluate import MemSumInference  # type: ignore

def extract_text_epub(epub_path: str) -> str:
    try:
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise RuntimeError("Faltan dependencias. Instala: pip install ebooklib beautifulsoup4") from e

    if not os.path.exists(epub_path):
        raise FileNotFoundError(f"No existe el archivo: {epub_path}")

    book = epub.read_epub(epub_path)
    texts = []
    for item in book.get_items():
        if item.get_type() == 9:  # DOCUMENT
            try:
                content = item.get_content().decode('utf-8', errors='ignore')
            except Exception:
                content = ''
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                # eliminar scripts/estilos
                for tag in soup(['script', 'style']):
                    tag.extract()
                txt = soup.get_text(separator=' ')
                if txt:
                    # limpieza ligera
                    texts.append(' '.join(txt.split()))
    return '\n'.join(texts)


def main():
    p = argparse.ArgumentParser(description='Resumir ePub con MemSum')
    p.add_argument('epub', type=str, help='Ruta al archivo .epub')
    p.add_argument('--checkpoint', type=str, default=os.path.join(ROOT, 'checkpoints', 'best_model.pt'))
    p.add_argument('--config', type=str, default=os.path.join(ROOT, 'configs', 'booksum_config.yaml'))
    p.add_argument('--lang', type=str, default='spanish')
    p.add_argument('--max_summary_len', type=int, default=8)
    p.add_argument('--strategy', type=str, default='greedy', choices=['greedy', 'sample'])
    p.add_argument('--redundancy_penalty', type=float, default=0.3)
    p.add_argument('--sbert_mmr', action='store_true')
    p.add_argument('--mmr_alpha', type=float, default=0.6)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    text = extract_text_epub(args.epub)
    if not text.strip():
        print('No se pudo extraer texto del ePub o está vacío.')
        return

    infer = MemSumInference(args.checkpoint, args.config)
    summary = infer.summarize_text(
        text,
        max_summary_length=args.max_summary_len,
        lang=args.lang,
        strategy=args.strategy,
        redundancy_penalty=args.redundancy_penalty,
        dedup=True,
        reorder_by_position=True,
        sbert_mmr=args.sbert_mmr,
        mmr_alpha=args.mmr_alpha,
    )

    print('\nResumen generado:\n')
    print(summary)

    if args.output:
        out_dir = os.path.dirname(args.output)
        os.makedirs(out_dir, exist_ok=True) if out_dir else None
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f'\nResumen guardado en: {args.output}')


if __name__ == '__main__':
    main()
