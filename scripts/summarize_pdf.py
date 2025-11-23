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
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np
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


def split_into_sentences(text: str) -> list:
    """Split text into sentences"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def calculate_coherence(summary: str, model: SentenceTransformer = None) -> float:
    """
    Calcula la coherencia del resumen midiendo la similitud semántica
    entre oraciones consecutivas.
    """
    sentences = split_into_sentences(summary)
    if len(sentences) < 2:
        return 1.0
    
    try:
        if model is None:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1])
            similarities.append(sim)
        
        return float(np.mean(similarities))
    except Exception:
        return 0.0

def calculate_cohesion(summary: str, model: SentenceTransformer = None) -> float:
    """
    Calcula la cohesión del resumen midiendo qué tan relacionadas están
    todas las oraciones con el tema central.
    """
    sentences = split_into_sentences(summary)
    if len(sentences) < 2:
        return 1.0
    
    try:
        if model is None:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        similarities = []
        for emb in embeddings:
            sim = np.dot(emb, centroid)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    except Exception:
        return 0.0

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
    parser.add_argument('--max_summary_len', type=int, default=12,
                        help='Número máximo de oraciones en el resumen (default: 12 para mejor coherencia)')
    parser.add_argument('--min_sent_tokens', type=int, default=4,
                        help='Mínimo de tokens para considerar una oración candidata')
    parser.add_argument('--no_clean', action='store_true',
                        help='No aplicar limpieza del texto extraído')
    parser.add_argument('--strategy', type=str, default='sample', choices=['greedy', 'sample'],
                        help='Estrategia de decodificación (default: sample para mejor flujo)')
    parser.add_argument('--redundancy_penalty', type=float, default=0.1,
                        help='Penalización por redundancia (default: 0.1 para mejor coherencia, 0.0 desactiva)')
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

    # Calculate ROUGE metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(text, summary)
    
    # Calculate coherence and cohesion
    print("\n⏳ Calculando métricas de coherencia y cohesión...")
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    coherence = calculate_coherence(summary, sbert_model)
    cohesion = calculate_cohesion(summary, sbert_model)
    
    # Salida
    print("\n" + "="*80)
    print("📝 RESUMEN GENERADO")
    print("="*80)
    print(summary)
    print("\n" + "="*80)
    print("📊 MÉTRICAS DE EVALUACIÓN")
    print("="*80)
    
    # Display metrics in a formatted table
    print(f"\n{'Métrica':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 53)
    print(f"{'ROUGE-1':<15} {rouge_scores['rouge1'].precision:<12.4f} {rouge_scores['rouge1'].recall:<12.4f} {rouge_scores['rouge1'].fmeasure:<12.4f}")
    print(f"{'ROUGE-2':<15} {rouge_scores['rouge2'].precision:<12.4f} {rouge_scores['rouge2'].recall:<12.4f} {rouge_scores['rouge2'].fmeasure:<12.4f}")
    print(f"{'ROUGE-L':<15} {rouge_scores['rougeL'].precision:<12.4f} {rouge_scores['rougeL'].recall:<12.4f} {rouge_scores['rougeL'].fmeasure:<12.4f}")
    print()
    print(f"{'🔗 Coherencia':<15} {'-':<12} {'-':<12} {coherence:<12.4f}")
    print(f"{'🎯 Cohesión':<15} {'-':<12} {'-':<12} {cohesion:<12.4f}")
    
    # Calculate additional statistics
    compression_ratio = (len(summary) / len(text) * 100) if len(text) > 0 else 0
    
    print("\n" + "="*80)
    print("📈 ESTADÍSTICAS")
    print("="*80)
    print(f"Texto original:      {len(text):>10,} caracteres")
    print(f"Resumen generado:    {len(summary):>10,} caracteres")
    print(f"Ratio de compresión: {compression_ratio:>10.2f}%")
    print(f"Oraciones en resumen:{args.max_summary_len:>10} (máx)")
    print("="*80)
    
    print("\n💡 Interpretación:")
    print("  • ROUGE-1: Similitud de palabras individuales (unigrams)")
    print("  • ROUGE-2: Similitud de pares de palabras (bigrams)")
    print("  • ROUGE-L: Secuencia común más larga")
    print("  • Coherencia: Conexión lógica entre oraciones consecutivas (>0.70 = buena)")
    print("  • Cohesión: Unidad temática del resumen (>0.75 = buena)")
    print("  • Precision: % del resumen que está en el original")
    print("  • Recall: % del original capturado en el resumen")
    print("  • F1-Score: Media armónica de Precision y Recall")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RESUMEN GENERADO\n")
            f.write("="*80 + "\n\n")
            f.write(summary)
            f.write("\n\n" + "="*80 + "\n")
            f.write("MÉTRICAS DE EVALUACIÓN (ROUGE)\n")
            f.write("="*80 + "\n\n")
            f.write(f"ROUGE-1 - F1: {rouge_scores['rouge1'].fmeasure:.4f} (P: {rouge_scores['rouge1'].precision:.4f}, R: {rouge_scores['rouge1'].recall:.4f})\n")
            f.write(f"ROUGE-2 - F1: {rouge_scores['rouge2'].fmeasure:.4f} (P: {rouge_scores['rouge2'].precision:.4f}, R: {rouge_scores['rouge2'].recall:.4f})\n")
            f.write(f"ROUGE-L - F1: {rouge_scores['rougeL'].fmeasure:.4f} (P: {rouge_scores['rougeL'].precision:.4f}, R: {rouge_scores['rougeL'].recall:.4f})\n")
            f.write(f"\nCoherencia: {coherence:.4f}\n")
            f.write(f"Cohesión: {cohesion:.4f}\n")
            f.write(f"\nRatio de compresión: {compression_ratio:.2f}%\n")
        print(f"\n✅ Resumen y métricas guardados en: {args.output}")


if __name__ == '__main__':
    main()
