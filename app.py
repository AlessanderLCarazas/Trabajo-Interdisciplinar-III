#!/usr/bin/env python3
"""
FastAPI REST API para MemSum - Resúmenes automáticos de PDFs
============================================================

Endpoints:
- POST /upload: Subir PDF y generar resumen
- GET /health: Verificar estado del servidor
- GET /: Interfaz web principal

Uso:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import logging

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from evaluate import MemSumInference
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MemSum PDF Summarizer",
    description="API REST para resumir PDFs usando MemSum con Reinforcement Learning",
    version="1.0.0"
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_instance: Optional[MemSumInference] = None

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
        logger.info(f"Model loaded successfully from {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def extract_text_from_pdf(pdf_path: str, extractor: str = "pdfminer") -> str:
    """Extract text from PDF using pdfminer or PyPDF2"""
    try:
        if extractor == "pdfminer":
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path) or ''
        else:
            import PyPDF2
            text_chunks = []
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    try:
                        txt = page.extract_text() or ''
                        if txt:
                            text_chunks.append(' '.join(txt.split()))
                    except Exception:
                        continue
            text = '\n'.join(text_chunks)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

# Global Sentence-BERT model for coherence/cohesion metrics
sbert_model = None

def load_sbert_model():
    """Load Sentence-BERT model for coherence metrics"""
    global sbert_model
    if sbert_model is None:
        sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return sbert_model

def split_into_sentences(text: str) -> list:
    """Split text into sentences"""
    # Simple sentence splitter
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def calculate_coherence(summary: str) -> float:
    """
    Calcula la coherencia del resumen midiendo la similitud semántica
    entre oraciones consecutivas.
    
    Coherencia alta = oraciones relacionadas entre sí
    Rango: 0.0 (no coherente) a 1.0 (muy coherente)
    """
    sentences = split_into_sentences(summary)
    if len(sentences) < 2:
        return 1.0  # Un solo sentence es perfectamente coherente
    
    try:
        model = load_sbert_model()
        embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        
        # Calcular similitud coseno entre oraciones consecutivas
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1])
            similarities.append(sim)
        
        coherence_score = float(np.mean(similarities))
        return max(0.0, min(1.0, coherence_score))
    
    except Exception as e:
        logger.warning(f"Error calculating coherence: {e}")
        return 0.0

def calculate_cohesion(summary: str) -> float:
    """
    Calcula la cohesión del resumen midiendo qué tan relacionadas están
    todas las oraciones con el tema central (promedio de embeddings).
    
    Cohesión alta = todas las oraciones tratan del mismo tema
    Rango: 0.0 (no cohesivo) a 1.0 (muy cohesivo)
    """
    sentences = split_into_sentences(summary)
    if len(sentences) < 2:
        return 1.0  # Un solo sentence es perfectamente cohesivo
    
    try:
        model = load_sbert_model()
        embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        
        # Calcular el centroide (tema central)
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalizar
        
        # Calcular similitud de cada oración con el centroide
        similarities = []
        for emb in embeddings:
            sim = np.dot(emb, centroid)
            similarities.append(sim)
        
        cohesion_score = float(np.mean(similarities))
        return max(0.0, min(1.0, cohesion_score))
    
    except Exception as e:
        logger.warning(f"Error calculating cohesion: {e}")
        return 0.0

@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    load_model()
    load_sbert_model()  # Pre-cargar modelo SBERT

@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MemSum - Resúmenes Automáticos de PDFs</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #1a1a1a;
                color: #ffffff;
                min-height: 100vh;
                overflow-x: hidden;
            }
            
            .header {
                background: #2a2a2a;
                color: #ffffff;
                padding: 20px;
                text-align: center;
                border-bottom: 2px solid #444;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                color: #00bcd4;
            }
            
            .header p {
                opacity: 0.8;
                font-size: 1.1em;
            }
            
            .main-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                min-height: calc(100vh - 140px);
                gap: 0;
            }
            
            .left-panel {
                background: #2a2a2a;
                padding: 40px;
                border-right: 2px solid #444;
            }
            
            .right-panel {
                background: #1f1f1f;
                padding: 40px;
                display: flex;
                flex-direction: column;
            }
            
            .upload-zone {
                border: 3px dashed #555;
                border-radius: 10px;
                padding: 60px 20px;
                text-align: center;
                background: #333;
                transition: all 0.3s ease;
                cursor: pointer;
                margin-bottom: 30px;
            }
            
            .upload-zone:hover {
                border-color: #00bcd4;
                background: #3a3a3a;
            }
            
            .upload-zone.dragover {
                border-color: #00bcd4;
                background: #404040;
                transform: scale(1.02);
            }
            
            .upload-icon {
                font-size: 3em;
                color: #00bcd4;
                margin-bottom: 20px;
                font-weight: bold;
            }
            
            .upload-text {
                font-size: 1.2em;
                margin-bottom: 10px;
                color: #ffffff;
            }
            
            .upload-hint {
                color: #aaa;
                font-size: 0.9em;
            }
            
            .controls {
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            
            .control-group {
                flex: 1;
                min-width: 200px;
            }
            
            .control-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #ffffff;
            }
            
            .control-group select, .control-group input {
                width: 100%;
                padding: 12px;
                border: 2px solid #555;
                border-radius: 5px;
                font-size: 1em;
                background: #444;
                color: #ffffff;
            }
            
            .control-group select:focus, .control-group input:focus {
                border-color: #00bcd4;
                outline: none;
            }
            
            .btn {
                background: linear-gradient(45deg, #00bcd4, #0097a7);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
                font-weight: 600;
            }
            
            .btn:hover {
                background: linear-gradient(45deg, #00acc1, #00838f);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 188, 212, 0.3);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
                background: #555;
            }
            
            .progress {
                width: 100%;
                height: 8px;
                background: #444;
                border-radius: 4px;
                overflow: hidden;
                margin: 20px 0;
                display: none;
            }
            
            .progress-bar {
                height: 100%;
                background: linear-gradient(45deg, #00bcd4, #0097a7);
                width: 0%;
                transition: width 0.3s ease;
            }
            
            .results {
                flex: 1;
                display: none;
            }
            
            .results h3 {
                color: #00bcd4;
                margin-bottom: 20px;
                font-size: 1.5em;
            }
            
            .summary-text {
                background: #333;
                padding: 25px;
                border-radius: 8px;
                border-left: 4px solid #00bcd4;
                line-height: 1.6;
                font-size: 1.1em;
                white-space: pre-wrap;
                color: #ffffff;
                max-height: 60vh;
                overflow-y: auto;
                margin-bottom: 20px;
            }
            
            .summary-text::-webkit-scrollbar {
                width: 8px;
            }
            
            .summary-text::-webkit-scrollbar-track {
                background: #444;
                border-radius: 4px;
            }
            
            .summary-text::-webkit-scrollbar-thumb {
                background: #00bcd4;
                border-radius: 4px;
            }
            
            .download-btn {
                background: #4CAF50;
                padding: 12px 25px;
                font-size: 1em;
                width: auto;
                align-self: flex-start;
            }
            
            .download-btn:hover {
                background: #45a049;
            }
            
            .error {
                background: #1b0000;
                color: #ff6b6b;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #ff3333;
                margin-top: 15px;
                display: none;
            }
            
            .footer {
                background: #2a2a2a;
                padding: 15px;
                text-align: center;
                color: #888;
                font-size: 0.9em;
                border-top: 2px solid #444;
            }
            
            .empty-state {
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: #666;
                font-size: 1.2em;
            }
            
            @media (max-width: 768px) {
                .main-container {
                    grid-template-columns: 1fr;
                    grid-template-rows: auto auto;
                }
                
                .left-panel {
                    border-right: none;
                    border-bottom: 2px solid #444;
                }
                
                .controls {
                    flex-direction: column;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .upload-zone {
                    padding: 40px 15px;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MemSum</h1>
            <p>Resúmenes de PDFs</p>
        </div>
        
        <div class="main-container">
            <div class="left-panel">
                <div class="upload-zone" id="uploadZone">
                    <div class="upload-icon">PDF</div>
                    <div class="upload-text">Arrastra tu PDF aquí o haz clic para seleccionar</div>
                    <div class="upload-hint">Archivos PDF hasta 10MB</div>
                </div>
                
                <input type="file" id="fileInput" accept=".pdf" style="display: none;">
                
                <div class="controls">
                    <div class="control-group">
                        <label for="language">Idioma del documento:</label>
                        <select id="language">
                            <option value="spanish">Español</option>
                            <option value="english">English</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label for="maxLength">Oraciones en resumen:</label>
                        <input type="number" id="maxLength" value="12" min="3" max="20">
                        <small style="color: #888; font-size: 0.85em;">Recomendado: 12 para mejor coherencia</small>
                    </div>
                </div>
                
                <button class="btn" id="summarizeBtn" disabled>
                    Generar Resumen
                </button>
                
                <div class="progress" id="progress">
                    <div class="progress-bar" id="progressBar"></div>
                </div>
                
                <div class="error" id="error"></div>
            </div>
            
            <div class="right-panel">
                <div class="empty-state" id="emptyState">
                    Selecciona un PDF para generar su resumen
                </div>
                
                <div class="results" id="results">
                    <h3>Resumen Generado:</h3>
                    <div class="summary-text" id="summaryText"></div>
                    <button class="btn download-btn" id="downloadBtn">
                        Descargar Resumen
                    </button>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Powered by MemSum + Reinforcement Learning</p>
        </div>
        
        <script>
            let selectedFile = null;
            let summaryResult = null;
            
            // DOM elements
            const uploadZone = document.getElementById('uploadZone');
            const fileInput = document.getElementById('fileInput');
            const summarizeBtn = document.getElementById('summarizeBtn');
            const progress = document.getElementById('progress');
            const progressBar = document.getElementById('progressBar');
            const results = document.getElementById('results');
            const summaryText = document.getElementById('summaryText');
            const downloadBtn = document.getElementById('downloadBtn');
            const errorDiv = document.getElementById('error');
            const languageSelect = document.getElementById('language');
            const maxLengthInput = document.getElementById('maxLength');
            const emptyState = document.getElementById('emptyState');
            
            // Upload zone events
            uploadZone.addEventListener('click', () => fileInput.click());
            uploadZone.addEventListener('dragover', handleDragOver);
            uploadZone.addEventListener('dragleave', handleDragLeave);
            uploadZone.addEventListener('drop', handleDrop);
            
            fileInput.addEventListener('change', handleFileSelect);
            summarizeBtn.addEventListener('click', generateSummary);
            downloadBtn.addEventListener('click', downloadSummary);
            
            function handleDragOver(e) {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            }
            
            function handleDragLeave(e) {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
            }
            
            function handleDrop(e) {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            }
            
            function handleFileSelect(e) {
                const file = e.target.files[0];
                if (file) {
                    handleFile(file);
                }
            }
            
            function handleFile(file) {
                if (!file.type.includes('pdf')) {
                    showError('Por favor selecciona un archivo PDF válido.');
                    return;
                }
                
                if (file.size > 10 * 1024 * 1024) {
                    showError('El archivo es demasiado grande. Máximo 10MB.');
                    return;
                }
                
                selectedFile = file;
                uploadZone.innerHTML = `
                    <div class="upload-icon">OK</div>
                    <div class="upload-text">Archivo seleccionado: ${file.name}</div>
                    <div class="upload-hint">${(file.size / 1024 / 1024).toFixed(2)} MB</div>
                `;
                
                summarizeBtn.disabled = false;
                hideError();
                hideResults();
            }
            
            async function generateSummary() {
                if (!selectedFile) return;
                
                summarizeBtn.disabled = true;
                showProgress();
                hideError();
                hideResults();
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('language', languageSelect.value);
                formData.append('max_summary_length', maxLengthInput.value);
                
                try {
                    updateProgress(30);
                    
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    updateProgress(70);
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Error procesando el archivo');
                    }
                    
                    const result = await response.json();
                    updateProgress(100);
                    
                    setTimeout(() => {
                        showResults(result);
                        hideProgress();
                    }, 500);
                    
                } catch (error) {
                    showError('Error: ' + error.message);
                    hideProgress();
                } finally {
                    summarizeBtn.disabled = false;
                }
            }
            
            function showProgress() {
                progress.style.display = 'block';
                updateProgress(10);
            }
            
            function hideProgress() {
                progress.style.display = 'none';
                updateProgress(0);
            }
            
            function updateProgress(percent) {
                progressBar.style.width = percent + '%';
            }
            
            function showResults(result) {
                summaryResult = result.summary;
                
                // Display summary
                summaryText.textContent = result.summary;
                
                // Display metrics (Coherence, Cohesion, Compression)
                const metricsHTML = `
                    <div style="margin-top: 20px; padding: 20px; background: #2a2a2a; border-radius: 10px; border: 1px solid #444;">
                        <h3 style="color: #00bcd4; margin-bottom: 15px;">📊 Métricas de Calidad</h3>
                        
                        <div style="display: flex; justify-content: space-between; padding: 15px; background: #1a1a1a; border-radius: 8px; margin-bottom: 15px;">
                            <div>
                                <span style="color: #888;">Texto original:</span>
                                <span style="color: #fff; margin-left: 10px; font-weight: bold;">${result.text_length.toLocaleString()} caracteres</span>
                            </div>
                            <div>
                                <span style="color: #888;">Resumen:</span>
                                <span style="color: #fff; margin-left: 10px; font-weight: bold;">${result.summary_length.toLocaleString()} caracteres</span>
                            </div>
                            <div>
                                <span style="color: #888;">Compresión:</span>
                                <span style="color: #00bcd4; margin-left: 10px; font-weight: bold;">${result.compression_ratio}%</span>
                            </div>
                        </div>
                        
                        ${result.coherence !== undefined && result.cohesion !== undefined ? `
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                                <div style="color: #888; font-size: 0.9em; margin-bottom: 5px;">🔗 Coherencia</div>
                                <div style="font-size: 1.8em; color: #9c27b0; font-weight: bold;">${(result.coherence * 100).toFixed(2)}%</div>
                                <div style="font-size: 0.75em; color: #aaa; margin-top: 5px;">
                                    Similitud entre oraciones consecutivas
                                </div>
                            </div>
                            
                            <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                                <div style="color: #888; font-size: 0.9em; margin-bottom: 5px;">🎯 Cohesión</div>
                                <div style="font-size: 1.8em; color: #e91e63; font-weight: bold;">${(result.cohesion * 100).toFixed(2)}%</div>
                                <div style="font-size: 0.75em; color: #aaa; margin-top: 5px;">
                                    Unidad temática del resumen
                                </div>
                            </div>
                        </div>
                        ` : ''}
                        
                        <div style="margin-top: 15px; padding: 10px; background: #1a1a1a; border-radius: 5px; font-size: 0.85em; color: #aaa;">
                            <strong>📖 Interpretación de Métricas:</strong>
                            <ul style="margin: 5px 0; padding-left: 20px;">
                                <li><strong>Compresión:</strong> Porcentaje del texto original que quedó en el resumen</li>
                                ${result.coherence !== undefined ? '<li><strong>Coherencia:</strong> Conexión lógica entre oraciones (>70% = buena)</li>' : ''}
                                ${result.cohesion !== undefined ? '<li><strong>Cohesión:</strong> Unidad temática del resumen (>75% = buena)</li>' : ''}
                            </ul>
                            <div style="margin-top: 10px; padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #00bcd4;">
                                <strong>ℹ️ Nota sobre ROUGE:</strong> Las métricas ROUGE (comparación con resumen humano) solo están disponibles 
                                en <code style="background: #1a1a1a; padding: 2px 6px; border-radius: 3px;">evaluate.py</code> cuando se evalúa con el dataset BookSum.
                            </div>
                        </div>
                    </div>
                `;
                
                // Insert metrics after summary text
                summaryText.insertAdjacentHTML('afterend', metricsHTML);
                
                emptyState.style.display = 'none';
                results.style.display = 'block';
            }
            
            function hideResults() {
                results.style.display = 'none';
                emptyState.style.display = 'flex';
            }
            
            function showError(message) {
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
            }
            
            function hideError() {
                errorDiv.style.display = 'none';
            }
            
            function downloadSummary() {
                if (!summaryResult) return;
                
                const blob = new Blob([summaryResult], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                
                a.href = url;
                a.download = `resumen_${new Date().toISOString().slice(0,10)}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                
                URL.revokeObjectURL(url);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "version": "1.0.0"
    }

@app.post("/upload")
async def upload_and_summarize(
    file: UploadFile = File(...),
    language: str = Form("spanish"),
    max_summary_length: int = Form(12),  # Aumentado de 8 a 12 para mejor coherencia
    strategy: str = Form("sample")  # Cambiado de greedy a sample para mejor flujo
):
    """Upload PDF and generate summary"""
    
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        try:
            # Save uploaded file
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
            
            # Extract text from PDF
            text = extract_text_from_pdf(tmp_path, extractor="pdfminer")
            
            if not text.strip():
                # Try fallback extractor
                text = extract_text_from_pdf(tmp_path, extractor="pypdf2")
            
            if not text.strip():
                raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")
            
            # Generate summary with optimized parameters for better coherence/cohesion
            summary = model_instance.summarize_text(
                text,
                max_summary_length=max_summary_length,
                lang=language,
                strategy=strategy,
                redundancy_penalty=0.1,  # Reducido de 0.3 a 0.1 para mejor coherencia
                dedup=True
            )
            
            # NOTA: ROUGE se ELIMINÓ de aquí porque requiere resumen humano de referencia
            # La interfaz web procesa PDFs arbitrarios sin resúmenes de referencia
            # ROUGE solo está disponible en evaluate.py cuando se usa dataset BookSum
            
            # Calculate coherence and cohesion
            coherence = round(calculate_coherence(summary), 4)
            cohesion = round(calculate_cohesion(summary), 4)
            
            # Calculate compression ratio
            compression_ratio = round(len(summary) / len(text) * 100, 2) if len(text) > 0 else 0
            
            return {
                "filename": file.filename,
                "summary": summary,
                "language": language,
                "max_length": max_summary_length,
                "text_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": compression_ratio,
                "coherence": coherence,
                "cohesion": cohesion,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )