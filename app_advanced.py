#!/usr/bin/env python3
"""
MemSum - Interfaz Web Avanzada con Dos Modos
=============================================
1. Modo PDF: Sube PDF, genera resumen (sin ROUGE, sin resumen humano)
2. Modo BookSum: Selecciona libro del dataset, genera resumen (con ROUGE, con resumen humano)

Uso:
    python app_advanced.py
    Abre: http://localhost:8000
"""

import os
import sys
import tempfile
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from evaluate import MemSumInference
from rouge_score import rouge_scorer
from datasets import load_dataset
from nltk import sent_tokenize
import nltk

# Asegurar punkt
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MemSum Advanced - PDF & BookSum",
    description="Resumir PDFs o libros del dataset BookSum con evaluación de métricas",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_instance: Optional[MemSumInference] = None
booksum_dataset = None
booksum_titles = []

def load_model():
    """Load MemSum model on startup"""
    global model_instance
    try:
        checkpoint_path = "checkpoints/best_model.pt"
        config_path = "configs/booksum_config.yaml"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        model_instance = MemSumInference(checkpoint_path, config_path)
        logger.info(f"✅ Model loaded successfully from {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def load_booksum_dataset():
    """Load BookSum dataset for book selection"""
    global booksum_dataset, booksum_titles
    try:
        logger.info("Loading BookSum dataset...")
        booksum_dataset = load_dataset('kmfoda/booksum', split='validation')
        
        # Extract book titles (primeros 100 para no saturar)
        booksum_titles = []
        seen_books = {}  # {book_name: first_index}
        
        for i, example in enumerate(booksum_dataset):
            if len(seen_books) >= 100:  # Limitar a 100 libros únicos
                break
            
            # Extraer nombre del libro desde book_id (ej: "Bleak House.chapter 1" -> "Bleak House")
            book_id = example.get('book_id', '')
            if book_id and '.' in book_id:
                book_name = book_id.split('.')[0].strip()
            else:
                book_name = f"Libro #{i+1}"
            
            # Solo agregar el primer capítulo de cada libro
            if book_name not in seen_books:
                seen_books[book_name] = i
                booksum_titles.append({
                    'index': i,
                    'title': book_name[:80]  # Truncar títulos muy largos
                })
        
        logger.info(f"✅ BookSum dataset loaded: {len(booksum_titles)} books available")
        
    except Exception as e:
        logger.error(f"❌ Failed to load BookSum dataset: {e}")
        booksum_titles = []

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfminer"""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path) or ''
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

@app.on_event("startup")
async def startup_event():
    """Load model and dataset on startup"""
    load_model()
    load_booksum_dataset()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with dual mode interface"""
    return HTMLResponse(content=get_html_interface())

@app.get("/api/books")
async def get_books():
    """Get list of available books from BookSum"""
    return {"books": booksum_titles}

@app.get("/api/book_preview")
async def get_book_preview(book_index: int):
    """Get preview of book text and human summary"""
    try:
        if booksum_dataset is None or book_index >= len(booksum_dataset):
            raise HTTPException(status_code=404, detail="Book not found")
        
        example = booksum_dataset[book_index]
        text = example.get('chapter') or example.get('text', '')
        summary = example.get('summary_text') or example.get('summary', '')
        
        return {
            "status": "success",
            "text": text,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error loading book preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize_pdf")
async def summarize_pdf(
    file: UploadFile = File(...),
    max_sentences: int = Form(10)
):
    """
    Modo 1: Resumir PDF subido
    Métricas: Solo compression ratio (sin ROUGE, sin coherence/cohesion)
    """
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Save uploaded file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract text
        text = extract_text_from_pdf(tmp_path)
        if not text or len(text) < 100:
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")
        
        # Generate summary
        summary = model_instance.summarize_text(
            text,
            max_summary_length=max_sentences,
            lang='en',
            strategy='greedy',
            redundancy_penalty=0.1,
            dedup=True
        )
        
        # Calculate basic metrics (no ROUGE, no coherence/cohesion)
        sentences_original = sent_tokenize(text)
        sentences_summary = sent_tokenize(summary)
        compression_ratio = round(len(summary) / len(text) * 100, 2) if len(text) > 0 else 0
        
        return {
            "mode": "pdf",
            "filename": file.filename,
            "original_text": text[:1000] + ("..." if len(text) > 1000 else ""),
            "summary": summary,
            "metrics": {
                "text_length": len(text),
                "summary_length": len(summary),
                "text_sentences": len(sentences_original),
                "summary_sentences": len(sentences_summary),
                "compression_ratio": compression_ratio
            },
            "note": "Métricas ROUGE no disponibles (sin resumen humano de referencia)",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.post("/api/summarize_book")
async def summarize_book(
    book_index: int = Form(...),
    max_sentences: int = Form(10)
):
    """
    Modo 2: Resumir libro de BookSum
    Métricas: ROUGE + BERTScore + Content Coverage (con resumen humano)
    """
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not booksum_dataset or book_index < 0 or book_index >= len(booksum_dataset):
        raise HTTPException(status_code=400, detail="Libro no encontrado")
    
    try:
        # Get book from dataset
        example = booksum_dataset[book_index]
        text = example.get('chapter', '') or example.get('text', '')
        human_summary = example.get('summary_text', '') or example.get('summary', '')
        title = example.get('book_title', '') or example.get('title', '') or f"Libro #{book_index+1}"
        
        if not text or not human_summary:
            raise HTTPException(status_code=400, detail="Libro sin texto o resumen")
        
        # Generate summary with model
        generated_summary = model_instance.summarize_text(
            text,
            max_summary_length=max_sentences,
            lang='en',
            strategy='greedy',
            redundancy_penalty=0.1,
            dedup=True
        )
        
        # Calculate ALL metrics (ROUGE, BERTScore, Content Coverage)
        metrics = calculate_all_metrics(generated_summary, human_summary, text)
        
        # Tokenizar para contar oraciones
        sentences_original = sent_tokenize(text)
        sentences_human = sent_tokenize(human_summary)
        sentences_generated = sent_tokenize(generated_summary)
        
        return {
            "mode": "booksum",
            "title": title,
            "original_text": text,
            "human_summary": human_summary,
            "generated_summary": generated_summary,
            "metrics": {
                **metrics,
                "text_sentences": len(sentences_original),
                "human_summary_sentences": len(sentences_human),
                "generated_summary_sentences": len(sentences_generated),
                "text_length": len(text),
                "human_summary_length": len(human_summary),
                "generated_summary_length": len(generated_summary),
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing book: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_all_metrics(generated_summary: str, human_summary: str, original_text: str) -> Dict[str, Any]:
    """
    Calcula TODAS las métricas de los papers:
    - ROUGE-1, 2, L (MemSum + BookSum)
    - BERTScore (BookSum)
    - Content Coverage (SummaQA-like from BookSum)
    """
    metrics = {}
    
    # 1. ROUGE
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(human_summary, generated_summary)
        
        metrics['rouge'] = {
            'rouge1': {
                'precision': round(rouge_scores['rouge1'].precision, 4),
                'recall': round(rouge_scores['rouge1'].recall, 4),
                'f1': round(rouge_scores['rouge1'].fmeasure, 4)
            },
            'rouge2': {
                'precision': round(rouge_scores['rouge2'].precision, 4),
                'recall': round(rouge_scores['rouge2'].recall, 4),
                'f1': round(rouge_scores['rouge2'].fmeasure, 4)
            },
            'rougeL': {
                'precision': round(rouge_scores['rougeL'].precision, 4),
                'recall': round(rouge_scores['rougeL'].recall, 4),
                'f1': round(rouge_scores['rougeL'].fmeasure, 4)
            }
        }
    except Exception as e:
        logger.error(f"Error calculating ROUGE: {e}")
        metrics['rouge'] = None
    
    # 2. BERTScore
    try:
        from bert_score import score as bert_score_func
        P, R, F1 = bert_score_func([generated_summary], [human_summary], lang='en', verbose=False)
        metrics['bertscore'] = {
            'precision': round(P.mean().item(), 4),
            'recall': round(R.mean().item(), 4),
            'f1': round(F1.mean().item(), 4)
        }
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {e}")
        metrics['bertscore'] = None
    
    # 3. Content Coverage (SummaQA-like)
    try:
        human_words = set(w.lower() for w in human_summary.split() if len(w) > 4)
        generated_words = set(w.lower() for w in generated_summary.split() if len(w) > 4)
        
        if human_words:
            coverage = len(human_words & generated_words) / len(human_words)
            metrics['content_coverage'] = round(coverage, 4)
        else:
            metrics['content_coverage'] = 0.0
    except Exception as e:
        logger.error(f"Error calculating Content Coverage: {e}")
        metrics['content_coverage'] = 0.0
    
    return metrics

def get_html_interface():
    """Generate HTML interface with dual mode"""
    return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MemSum - Resúmenes Extractivos</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .mode-selector {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            justify-content: center;
        }
        
        .mode-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px 40px;
            cursor: pointer;
            transition: all 0.3s;
            border: 2px solid rgba(255,255,255,0.2);
        }
        
        .mode-card:hover {
            background: rgba(255,255,255,0.15);
            transform: translateY(-2px);
        }
        
        .mode-card.active {
            background: rgba(102, 126, 234, 0.3);
            border-color: #667eea;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.5);
        }
        
        .mode-card h2 {
            color: white;
            margin-bottom: 5px;
            font-size: 1.5em;
        }
        
        .content-panel {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 30px;
            display: none;
        }
        
        .content-panel.active {
            display: block;
        }
        
        .two-column-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .column {
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .column h3 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 10px;
            color: rgba(255,255,255,0.9);
            font-weight: 600;
        }
        
        .form-group input[type="file"],
        .form-group select,
        .form-group input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            font-size: 1em;
            background: rgba(0,0,0,0.3);
            color: white;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(0,0,0,0.4);
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.6);
        }
        
        .btn:disabled {
            background: #555;
            cursor: not-allowed;
            transform: none;
        }
        
        .preview-box {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .preview-box h4 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .preview-text {
            color: rgba(255,255,255,0.8);
            line-height: 1.6;
            font-size: 0.95em;
        }
        
        .summary-box {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #667eea;
        }
        
        .summary-box h4 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .summary-text {
            color: rgba(255,255,255,0.9);
            line-height: 1.8;
        }
        
        .metrics-section {
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 25px;
            margin-top: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .metrics-section h3 {
            color: #667eea;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric-label {
            color: rgba(255,255,255,0.7);
            font-size: 0.9em;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-sublabel {
            font-size: 0.8em;
            color: rgba(255,255,255,0.5);
            margin-top: 5px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        select option {
            background: #2d2d2d;
            color: white;
        }

        
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>

</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 MemSum</h1>
            <p>Resúmenes Extractivos</p>
        </div>
        
        <div class="mode-selector">
            <div class="mode-card" id="mode-pdf" onclick="selectMode('pdf')">
                <h2>📄 Resumir PDF</h2>
                <p style="margin-top: 10px; opacity: 0.8; font-size: 0.9em;">Sube un PDF y genera un resumen automático.</p>
            </div>
            
            <div class="mode-card" id="mode-booksum" onclick="selectMode('booksum')">
                <h2>📖 Resumir Libro (BookSum)</h2>
                <p style="margin-top: 10px; opacity: 0.8; font-size: 0.9em;">Selecciona un libro del dataset BookSum.</p>
            </div>
        </div>
        
        <!-- Panel PDF -->
        <div class="content-panel" id="panel-pdf">
            <div class="two-column-layout">
                <!-- Columna 1: Input -->
                <div class="column">
                    <h3>Configuración</h3>
                    <form id="form-pdf">
                        <div class="form-group">
                            <label>Seleccionar PDF:</label>
                            <input type="file" id="pdf-file" accept=".pdf" required onchange="previewPDF()">
                        </div>
                        
                        <div class="form-group">
                            <label>Número de oraciones:</label>
                            <input type="number" id="pdf-sentences" min="3" max="20" value="10" required>
                        </div>
                        
                        <button type="submit" class="btn" id="btn-pdf">Generar Resumen</button>
                    </form>
                    
                    <div class="loading" id="loading-pdf">
                        <div class="spinner"></div>
                        <p>Procesando...</p>
                    </div>
                    
                    <!-- Previsualización del texto extraído del PDF -->
                    <div id="pdf-preview" class="preview-box" style="display:none; margin-top: 20px;">
                        <h4>Previsualización del Texto Extraído</h4>
                        <div id="pdf-text" class="preview-text"></div>
                    </div>
                </div>
                
                <!-- Columna 2: Resumen -->
                <div class="column">
                    <div id="pdf-summary-container"></div>
                </div>
            </div>
            
            <!-- Métricas debajo de ambas columnas -->
            <div id="pdf-metrics-container"></div>
        </div>
        
        <!-- Panel BookSum -->
        <div class="content-panel" id="panel-booksum">
            <div class="two-column-layout">
                <!-- Columna 1: Selección y previsualización -->
                <div class="column">
                    <h3>Seleccionar Libro</h3>
                    <div class="form-group">
                        <label>Libro:</label>
                        <select id="book-select" onchange="showBookPreview()" required>
                            <option value="">Cargando libros...</option>
                        </select>
                    </div>
                    
                    <div id="book-preview" class="preview-box" style="display:none;">
                        <h4>Previsualización del Texto Original</h4>
                        <div id="book-text" class="preview-text"></div>
                    </div>
                    
                    <div id="book-human-summary" class="preview-box" style="display:none; margin-top: 15px;">
                        <h4>Resumen Humano (Referencia)</h4>
                        <div id="human-summary-text" class="preview-text"></div>
                    </div>
                </div>
                
                <!-- Columna 2: Configuración y resumen generado -->
                <div class="column">
                    <h3>Generar Resumen</h3>
                    <form id="form-booksum">
                        <div class="form-group">
                            <label>Número de oraciones:</label>
                            <input type="number" id="book-sentences" min="3" max="20" value="10" required>
                        </div>
                        
                        <button type="submit" class="btn" id="btn-booksum">Generar Resumen</button>
                    </form>
                    
                    <div class="loading" id="loading-booksum">
                        <div class="spinner"></div>
                        <p>Generando resumen...</p>
                    </div>
                    
                    <div id="booksum-summary-container"></div>
                </div>
            </div>
            
            <!-- Métricas debajo de ambas columnas -->
            <div id="booksum-metrics-container"></div>
        </div>
    </div>
    
    <script>
        let currentMode = null;
        let booksData = [];
        
        // Cargar libros al iniciar
        async function loadBooks() {
            try {
                const response = await fetch('/api/books');
                const data = await response.json();
                booksData = data.books;
                const select = document.getElementById('book-select');
                select.innerHTML = '<option value="">-- Selecciona un libro --</option>';
                data.books.forEach(book => {
                    const option = document.createElement('option');
                    option.value = book.index;
                    option.textContent = book.title;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading books:', error);
                document.getElementById('book-select').innerHTML = '<option value="">Error cargando libros</option>';
            }
        }
        
        loadBooks();
        
        function selectMode(mode) {
            currentMode = mode;
            
            // Update mode cards
            document.querySelectorAll('.mode-card').forEach(card => {
                card.classList.remove('active');
            });
            document.getElementById(`mode-${mode}`).classList.add('active');
            
            // Update panels
            document.querySelectorAll('.content-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            document.getElementById(`panel-${mode}`).classList.add('active');
        }
        
        // Función para mostrar previsualización del libro seleccionado
        async function showBookPreview() {
            const selectElement = document.getElementById('book-select');
            const bookIndex = selectElement.value;
            
            if (!bookIndex) {
                document.getElementById('book-preview').style.display = 'none';
                document.getElementById('book-human-summary').style.display = 'none';
                return;
            }
            
            try {
                const response = await fetch(`/api/book_preview?book_index=${bookIndex}`);
                const data = await response.json();
                
                if (data.status === 'success') {
                    // Mostrar texto original
                    document.getElementById('book-text').textContent = 
                        data.text.substring(0, 1500) + '... [texto completo se usará para el resumen]';
                    document.getElementById('book-preview').style.display = 'block';
                    
                    // Mostrar resumen humano
                    document.getElementById('human-summary-text').textContent = data.summary;
                    document.getElementById('book-human-summary').style.display = 'block';
                }
            } catch (error) {
                console.error('Error loading book preview:', error);
            }
        }
        
        // Previsualizar PDF cuando se selecciona
        function previewPDF() {
            const fileInput = document.getElementById('pdf-file');
            const file = fileInput.files[0];
            
            if (!file) {
                document.getElementById('pdf-preview').style.display = 'none';
                return;
            }
            
            // Mostrar nombre del archivo
            document.getElementById('pdf-text').textContent = 
                `Archivo seleccionado: ${file.name}\nTamaño: ${(file.size / 1024).toFixed(2)} KB\n\nHaz clic en "Generar Resumen" para extraer y resumir el texto.`;
            document.getElementById('pdf-preview').style.display = 'block';
        }
        
        // Handle PDF form
        document.getElementById('form-pdf').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('pdf-file');
            const sentences = document.getElementById('pdf-sentences').value;
            
            if (!fileInput.files[0]) {
                alert('Por favor selecciona un PDF');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('max_sentences', sentences);
            
            document.getElementById('loading-pdf').classList.add('active');
            document.getElementById('btn-pdf').disabled = true;
            
            try {
                const response = await fetch('/api/summarize_pdf', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    displayPDFResults(result);
                } else {
                    alert('Error: ' + (result.detail || 'Error desconocido'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading-pdf').classList.remove('active');
                document.getElementById('btn-pdf').disabled = false;
            }
        });
        
        // Handle BookSum form
        document.getElementById('form-booksum').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const bookIndex = document.getElementById('book-select').value;
            const sentences = document.getElementById('book-sentences').value;
            
            if (!bookIndex) {
                alert('Por favor selecciona un libro');
                return;
            }
            
            const formData = new FormData();
            formData.append('book_index', bookIndex);
            formData.append('max_sentences', sentences);
            
            document.getElementById('loading-booksum').classList.add('active');
            document.getElementById('btn-booksum').disabled = true;
            document.getElementById('btn-booksum').disabled = true;
            
            try {
                const response = await fetch('/api/summarize_book', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    displayBookSumResults(result);
                } else {
                    alert('Error: ' + (result.detail || 'Error desconocido'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading-booksum').classList.remove('active');
                document.getElementById('btn-booksum').disabled = false;
            }
        });
        
        function displayPDFResults(result) {
            const m = result.metrics;
            
            // Actualizar previsualización con texto extraído
            if (result.original_text) {
                document.getElementById('pdf-text').textContent = result.original_text;
                document.getElementById('pdf-preview').style.display = 'block';
            }
            
            // Mostrar resumen en columna 2
            document.getElementById('pdf-summary-container').innerHTML = `
                <h3 style="color: #667eea; margin-bottom: 20px;">Resumen Generado</h3>
                <div class="summary-box">
                    <div class="summary-text">${result.summary}</div>
                </div>
            `;
            
            // Mostrar métricas debajo
            document.getElementById('pdf-metrics-container').innerHTML = `
                <div class="metrics-section">
                    <h3>Métricas</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Oraciones Originales</div>
                            <div class="metric-value">${m.text_sentences}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Oraciones Resumen</div>
                            <div class="metric-value">${m.summary_sentences}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Ratio Compresión</div>
                            <div class="metric-value">${m.compression_ratio}%</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function displayBookSumResults(result) {
            const m = result.metrics;
            const rouge = m.rouge;
            const bert = m.bertscore;
            
            // Mostrar resumen en columna 2
            document.getElementById('booksum-summary-container').innerHTML = `
                <h3 style="color: #667eea; margin-bottom: 20px;">Resumen Generado (MemSum)</h3>
                <div class="summary-box">
                    <div class="summary-text">${result.generated_summary}</div>
                </div>
            `;
            
            // Mostrar métricas debajo de ambas columnas
            let metricsHTML = '<div class="metrics-section"><h3>Métricas de Evaluación</h3>';
            
            // ROUGE metrics
            if (rouge) {
                metricsHTML += `
                    <h4 style="color: #667eea; margin: 20px 0 15px;">ROUGE (MemSum + BookSum Papers)</h4>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">ROUGE-1</div>
                            <div class="metric-value">${(rouge.rouge1.f1 * 100).toFixed(2)}%</div>
                            <div class="metric-sublabel">P: ${(rouge.rouge1.precision * 100).toFixed(1)}% | R: ${(rouge.rouge1.recall * 100).toFixed(1)}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">ROUGE-2</div>
                            <div class="metric-value">${(rouge.rouge2.f1 * 100).toFixed(2)}%</div>
                            <div class="metric-sublabel">P: ${(rouge.rouge2.precision * 100).toFixed(1)}% | R: ${(rouge.rouge2.recall * 100).toFixed(1)}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">ROUGE-L</div>
                            <div class="metric-value">${(rouge.rougeL.f1 * 100).toFixed(2)}%</div>
                            <div class="metric-sublabel">P: ${(rouge.rougeL.precision * 100).toFixed(1)}% | R: ${(rouge.rougeL.recall * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                `;
            }
            
            // BERTScore
            if (bert) {
                metricsHTML += `
                    <h4 style="color: #667eea; margin: 20px 0 15px;">BERTScore (BookSum Paper)</h4>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Precision</div>
                            <div class="metric-value">${(bert.precision * 100).toFixed(2)}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Recall</div>
                            <div class="metric-value">${(bert.recall * 100).toFixed(2)}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">F1</div>
                            <div class="metric-value">${(bert.f1 * 100).toFixed(2)}%</div>
                        </div>
                    </div>
                `;
            }
            
            // Content Coverage
            if (m.content_coverage !== undefined) {
                metricsHTML += `
                    <h4 style="color: #667eea; margin: 20px 0 15px;">Content Coverage / SummaQA-like (BookSum Paper)</h4>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Coverage Score</div>
                            <div class="metric-value">${(m.content_coverage * 100).toFixed(2)}%</div>
                            <div class="metric-sublabel">Keyword overlap con resumen humano</div>
                        </div>
                    </div>
                `;
            }
            
            metricsHTML += '</div>';
            document.getElementById('booksum-metrics-container').innerHTML = metricsHTML;
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(
        "app_advanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
