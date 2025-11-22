#!/usr/bin/env python3
"""
Summarize a PDF file using MemSum checkpoint.
- Extracts text from PDF (PyPDF2)
- Splits into sentences
- Uses MemSumInference to generate an extractive summary
"""
import os
import sys
import argparse

# Ensure src on path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluate import MemSumInference
import re


def extract_text_pypdf2(pdf_path: str) -> str:
    try:
        import PyPDF2
    except ImportError as e:
        raise RuntimeError("PyPDF2 no está instalado. Instálalo con pip install PyPDF2") from e

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No existe el archivo: {pdf_path}")

    text_chunks = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ''
            except Exception:
                txt = ''
            if txt:
                # Normalización simple de espacios
                text_chunks.append(' '.join(txt.split()))
    return '\n'.join(text_chunks)


def extract_text_pdfminer(pdf_path: str) -> str:
    try:
        from pdfminer.high_level import extract_text
    except ImportError as e:
        raise RuntimeError("pdfminer.six no está instalado. Instálalo con pip install pdfminer.six") from e
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No existe el archivo: {pdf_path}")
    try:
        text = extract_text(pdf_path) or ''
        return text
    except Exception as e:
        # Fallback vacío si falla
        return ''


def clean_text(text: str) -> str:
    """Limpieza ligera del texto extraído del PDF.
    - Elimina líneas con solo números/simbolitos (p.ej. números de página)
    - Colapsa espacios múltiples
    - Normaliza guiones de diálogo y espacios alrededor de puntuación
    """
    # Quitar líneas vacías o con solo dígitos/puntuación breve
    lines = text.splitlines()
    keep = []
    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue
        # si es muy corta y sin letras (p. ej. "8")
        if len(raw) <= 3 and not re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", raw):
            continue
        keep.append(raw)
    text = " \n".join(keep)
    # Normalizar espacios
    text = re.sub(r"\s+", " ", text)
    # Espacios antes de puntuación
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Normalizar guiones largos de diálogo
    text = re.sub(r"\s*—\s*", " — ", text)
    # Evitar repetir guiones
    text = re.sub(r"\s*-\s*-\s*", " - ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description='Resumir un PDF con MemSum')
    parser.add_argument('pdf', type=str, help='Ruta al archivo PDF')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Ruta al checkpoint del modelo')
    parser.add_argument('--config', type=str, default='configs/booksum_config.yaml',
                        help='Ruta al archivo de configuración')
    parser.add_argument('--max_summary_len', type=int, default=8,
                        help='Número máximo de oraciones en el resumen')
    parser.add_argument('--min_sent_tokens', type=int, default=4,
                        help='Mínimo de tokens para considerar una oración candidata')
    parser.add_argument('--no_clean', action='store_true',
                        help='No aplicar limpieza del texto extraído')
    parser.add_argument('--strategy', type=str, default='greedy', choices=['greedy', 'sample'],
                        help='Estrategia de decodificación para seleccionar oraciones')
    parser.add_argument('--redundancy_penalty', type=float, default=0.3,
                        help='Penalización por redundancia (0.0 desactiva)')
    parser.add_argument('--no_dedup', action='store_true',
                        help='Desactiva la deduplicación de oraciones similares')
    parser.add_argument('--extractor', type=str, default='pdfminer', choices=['pdfminer','pypdf2'],
                        help='Método para extraer texto del PDF (pdfminer recomendado)')
    parser.add_argument('--output', type=str, default=None,
                        help='Archivo donde guardar el resumen (opcional)')
    parser.add_argument('--lang', type=str, default='spanish',
                        help='Idioma para segmentar oraciones (ej: spanish, english)')
    args = parser.parse_args()

    # Extraer texto
    # Elegir extractor
    if args.extractor == 'pdfminer':
        text = extract_text_pdfminer(args.pdf)
        if not text.strip():
            # Fallback a PyPDF2 si pdfminer no devolvió nada
            text = extract_text_pypdf2(args.pdf)
    else:
        text = extract_text_pypdf2(args.pdf)
    if not args.no_clean:
        text = clean_text(text)
    if not text.strip():
        print("No se pudo extraer texto del PDF o está vacío.")
        return

    # Crear inferencia y resumir
    infer = MemSumInference(args.checkpoint, args.config)
    # Filtrar oraciones demasiado cortas antes de resumir
    # Nota: el filtrado fino se hará en la tokenización de evaluate; aquí solo un pre-filtro simple
    summary = infer.summarize_text(
        text,
        max_summary_length=args.max_summary_len,
        lang=args.lang,
        strategy=args.strategy,
        redundancy_penalty=args.redundancy_penalty,
        dedup=not args.no_dedup,
    )

    # Salida
    print("\nResumen generado:\n")
    print(summary)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
        with open(args.output, 'w') as f:
            f.write(summary)
        print(f"\nResumen guardado en: {args.output}")


if __name__ == '__main__':
    main()
