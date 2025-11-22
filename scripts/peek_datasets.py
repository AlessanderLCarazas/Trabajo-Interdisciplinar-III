#!/usr/bin/env python3
"""
Muestra ejemplos del dataset BookSum (crudo) y del dataset procesado por nuestro pipeline.
- Crudo: directamente desde HuggingFace (kmfoda/booksum)
- Procesado: a través de src/data_loader.BookSumDataset (tokenizado + oráculo)

Uso:
  python scripts/peek_datasets.py --split validation --index 0 --max_docs 10 --truncate 400
"""
import os
import sys
import argparse

# Asegurar import de src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from datasets import load_dataset

try:
    from src.config import Config
    from src.data_loader import BookSumDataset
except ModuleNotFoundError:
    # Fallback: cargar por ruta directa
    from importlib.machinery import SourceFileLoader
    config_path = os.path.join(ROOT, 'src', 'config.py')
    dl_path = os.path.join(ROOT, 'src', 'data_loader.py')
    Config = SourceFileLoader('memsum_config', config_path).load_module().Config
    BookSumDataset = SourceFileLoader('memsum_dl', dl_path).load_module().BookSumDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', type=str, default='validation', choices=['train', 'validation', 'test'])
    p.add_argument('--index', type=int, default=0, help='Índice del ejemplo a mostrar')
    p.add_argument('--max_docs', type=int, default=20, help='Límite superior de documentos a procesar (para rapidez)')
    p.add_argument('--truncate', type=int, default=400, help='Máx. caracteres a imprimir por campo')
    return p.parse_args()


def truncate_text(text: str, n: int) -> str:
    text = text or ''
    return text[:n] + ('…' if len(text) > n else '')


def show_raw_booksum(split: str, idx: int, trunc: int, fout=None):
    print('=' * 80)
    print(f'[RAW] BookSum ({split}) ejemplo idx={idx}')
    if fout:
        fout.write('=' * 80 + '\n')
        fout.write(f'[RAW] BookSum ({split}) ejemplo idx={idx}\n')
    ds = load_dataset('kmfoda/booksum', split=split)
    if idx < 0 or idx >= len(ds):
        print(f'Índice fuera de rango (0..{len(ds)-1})')
        if fout:
            fout.write(f'Índice fuera de rango (0..{len(ds)-1})\n')
        return None
    ex = ds[idx]
    # Campos posibles en BookSum: 'chapter', 'text', 'summary_text', 'summary'
    chapter = ex.get('chapter') or ex.get('text') or ''
    summary = ex.get('summary_text') or ex.get('summary') or ''
    title = ex.get('book_title') or ex.get('title') or ''
    line_title = f'- Título (si disponible): {truncate_text(title, trunc)}'
    line_chap = f'- Capítulo/Text (trunc):\n{truncate_text(chapter, trunc)}'
    line_sum = f'- Resumen (trunc):\n{truncate_text(summary, trunc)}'
    print(line_title)
    print(line_chap)
    print(line_sum)
    if fout:
        fout.write(line_title + '\n')
        fout.write(line_chap + '\n')
        fout.write(line_sum + '\n')
    return {'chapter': chapter, 'summary': summary, 'title': title}


def show_processed_dataset(split: str, idx: int, max_docs: int, trunc: int, fout=None):
    print('=' * 80)
    print(f'[PROCESADO] BookSumDataset ({split}) ejemplo idx={idx} (max_docs={max_docs})')
    if fout:
        fout.write('=' * 80 + '\n')
        fout.write(f'[PROCESADO] BookSumDataset ({split}) ejemplo idx={idx} (max_docs={max_docs})\n')
    config_path = os.path.join(ROOT, 'configs', 'booksum_config.yaml')
    cfg = Config(config_path if os.path.exists(config_path) else None)
    ds = BookSumDataset(cfg, split=split, max_docs=max_docs)
    if idx < 0 or idx >= len(ds):
        print(f'Índice fuera de rango en procesado (0..{len(ds)-1})')
        if fout:
            fout.write(f'Índice fuera de rango en procesado (0..{len(ds)-1})\n')
        return None
    item = ds[idx]
    # Campos: sentences (idxs), summary (idxs), oracle_indices, raw_sentences, raw_summary
    l1 = f'- #oraciones documento: {len(item["raw_sentences"])}'
    l2 = f'- #oraciones resumen gold: {len(item["raw_summary"])}'
    l3 = f'- Oracle extractivo (índices): {item["oracle_indices"][:20]}'
    print(l1)
    print(l2)
    print(l3)
    if fout:
        fout.write(l1 + '\n')
        fout.write(l2 + '\n')
        fout.write(l3 + '\n')
    # Mostrar 2-3 oraciones del documento y resumen
    print('> Oraciones del documento (trunc):')
    if fout:
        fout.write('> Oraciones del documento (trunc):\n')
    for j, s in enumerate(item['raw_sentences'][:3]):
        line = f'  [{j}] {truncate_text(s, trunc)}'
        print(line)
        if fout:
            fout.write(line + '\n')
    print('> Oraciones del resumen (trunc):')
    if fout:
        fout.write('> Oraciones del resumen (trunc):\n')
    for j, s in enumerate(item['raw_summary'][:3]):
        line = f'  [{j}] {truncate_text(s, trunc)}'
        print(line)
        if fout:
            fout.write(line + '\n')
    return item


def main():
    args = get_args()
    # Además de imprimir por pantalla, guardamos en archivo
    out_dir = os.path.join(ROOT, 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'peek_{args.split}_idx{args.index}.txt')
    with open(out_path, 'w') as fout:
        _ = show_raw_booksum(args.split, args.index, args.truncate, fout)
        _ = show_processed_dataset(args.split, args.index, args.max_docs, args.truncate, fout)
    print(f"\n[OK] Vista guardada en: {out_path}")


if __name__ == '__main__':
    main()
