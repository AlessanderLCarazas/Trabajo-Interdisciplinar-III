# 📚 PROYECTO MEMSUM - Resúmenes Extractivos con RL

Implementación de **MemSum** (Memory-based Summarization) usando **Reinforcement Learning** para generar resúmenes extractivos de documentos largos.

---

## 🎯 ¿QUÉ HACE ESTE PROYECTO?

Genera **resúmenes automáticos** de textos largos (libros, PDFs, artículos) seleccionando las oraciones más importantes usando:
- 🧠 **Deep Learning** (PyTorch)
- 🎮 **Reinforcement Learning** (REINFORCE algorithm)
- 📊 **Evaluación con métricas** (ROUGE, BERTScore, Coverage)

---

## 📁 ESTRUCTURA DEL PROYECTO

```
jalar/
│
├── 📝 ARCHIVOS PRINCIPALES
│   ├── train.py              ← Entrenar el modelo desde cero
│   ├── evaluate.py           ← Evaluar modelo con métricas
│   ├── app.py                ← Interfaz web básica (solo PDFs)
│   ├── app_advanced.py       ← Interfaz web completa (PDF + BookSum)
│   └── requirements.txt      ← Dependencias a instalar
│
├── 🧠 CÓDIGO FUENTE (src/)
│   ├── model.py              ← Arquitectura MemSum (4 capas)
│   ├── trainer.py            ← Lógica de entrenamiento con RL
│   ├── data_loader.py        ← Carga dataset BookSum de HuggingFace
│   ├── config.py             ← Gestión de configuración
│   ├── fusion.py             ← Capas de fusión multimodal
│   └── __init__.py           ← Inicializador del paquete
│
├── ⚙️ CONFIGURACIÓN (configs/)
│   ├── booksum_config.yaml   ← Config para pruebas rápidas (5 epochs)
│   └── booksum_full_config.yaml ← Config para entrenamiento completo
│
├── 🤖 MODELO ENTRENADO (checkpoints/)
│   └── best_model.pt         ← Modelo ya entrenado (14 MB) ✅
│
├── 📊 DATOS (data/)
│   └── vocab.pkl             ← Vocabulario procesado
│
├── 🛠️ SCRIPTS AUXILIARES (scripts/)
│   ├── summarize_pdf.py      ← Resumir archivos PDF
│   └── summarize_epub.py     ← Resumir libros EPUB
│
└── 📖 DOCUMENTACIÓN
    ├── SETUP.md              ← Guía de instalación paso a paso
    ├── CONTENIDO.txt         ← Checklist de archivos incluidos
    ├── COMANDOS.txt          ← Lista de comandos importantes ⭐
    └── verificar_setup.py    ← Script para verificar instalación
```

---

## 🔍 ¿QUÉ HACE CADA ARCHIVO?

### 📝 Scripts Principales

#### `train.py`
**Para qué sirve**: Entrenar el modelo desde cero con el dataset BookSum.
- Usa Reinforcement Learning para aprender a seleccionar oraciones
- Guarda checkpoints cada epoch
- Funciona en CPU y GPU automáticamente

#### `evaluate.py`
**Para qué sirve**: Evaluar la calidad del modelo entrenado.
- Calcula métricas: ROUGE-1/2/L, BERTScore, Content Coverage
- Compara resúmenes generados vs resúmenes humanos
- Genera reporte de resultados

#### `app.py`
**Para qué sirve**: Interfaz web básica para resumir PDFs.
- Solo modo PDF (sin métricas completas)
- Rápido y sencillo
- Puerto 8000

#### `app_advanced.py`
**Para qué sirve**: Interfaz web completa con 2 modos.
- **Modo PDF**: Sube PDFs y genera resúmenes
- **Modo BookSum**: Selecciona libros y ve todas las métricas
- Visualización de ROUGE, BERTScore y Coverage
- Puerto 8000

---

### 🧠 Código Fuente (src/)

#### `model.py`
**Para qué sirve**: Define la arquitectura del modelo MemSum.
- **4 capas principales**:
  1. Sentence Encoder (LSTM bidireccional)
  2. Document Encoder (LSTM bidireccional con atención)
  3. Memory (almacena contexto de oraciones seleccionadas)
  4. Decoder (decide qué oración seleccionar)
- Usa embeddings GloVe o Word2Vec

#### `trainer.py`
**Para qué sirve**: Lógica de entrenamiento con Reinforcement Learning.
- Algoritmo REINFORCE para optimizar la selección
- Calcula rewards basados en ROUGE
- Maneja checkpoints y early stopping
- Muestra progreso y pérdida

#### `data_loader.py`
**Para qué sirve**: Carga y preprocesa el dataset BookSum.
- Descarga automáticamente desde HuggingFace
- Tokeniza textos y resúmenes
- Crea batches para entrenamiento
- Filtra documentos muy largos

#### `config.py`
**Para qué sirve**: Gestiona la configuración del proyecto.
- Lee archivos YAML
- Define hiperparámetros por defecto
- Valida configuración

#### `fusion.py`
**Para qué sirve**: Capas de fusión para combinar información.
- Fusiona representaciones de diferentes niveles
- Usado en el document encoder

---

### ⚙️ Configuración (configs/)

#### `booksum_config.yaml`
**Para qué sirve**: Configuración para entrenamiento rápido.
- 5 epochs (para pruebas)
- Batch size 2
- Para verificar que todo funciona

#### `booksum_full_config.yaml`
**Para qué sirve**: Configuración para entrenamiento completo.
- 40 epochs (entrenamiento serio)
- Mejores resultados
- Más tiempo de entrenamiento

**Parámetros que puedes ajustar**:
- `epochs`: Número de vueltas al dataset
- `batch_size`: Documentos por batch (1-8)
- `learning_rate`: Velocidad de aprendizaje
- `hidden_dim`: Tamaño de capas ocultas
- `num_layers`: Capas en LSTM

---

### 🤖 Modelo Entrenado (checkpoints/)

#### `best_model.pt`
**Para qué sirve**: Modelo ya entrenado listo para usar.
- 14 MB de tamaño
- Entrenado con BookSum dataset
- Funciona en CPU y GPU
- **¡No necesitas entrenar desde cero!**

---

### 🛠️ Scripts Auxiliares (scripts/)

#### `summarize_pdf.py`
**Para qué sirve**: Resume un PDF desde terminal.
- Extrae texto del PDF
- Genera resumen con el modelo
- Guarda resultado en archivo TXT

#### `summarize_epub.py`
**Para qué sirve**: Resume un libro EPUB desde terminal.
- Extrae texto del EPUB
- Genera resumen con el modelo
- Guarda resultado en archivo TXT

---

## 🚀 INICIO RÁPIDO

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Verificar instalación
python verificar_setup.py

# 3. Levantar interfaz web
python app_advanced.py

# Listo! Ve a: http://localhost:8000
```

---

## 📊 MÉTRICAS DE EVALUACIÓN

El proyecto evalúa resúmenes con:

| Métrica | Para qué sirve | Paper |
|---------|----------------|-------|
| **ROUGE-1** | Coincidencia de palabras individuales | MemSum + BookSum |
| **ROUGE-2** | Coincidencia de pares de palabras (fluidez) | MemSum + BookSum |
| **ROUGE-L** | Secuencia más larga común (estructura) | MemSum + BookSum |
| **BERTScore** | Similitud semántica (significado) | BookSum |
| **Content Coverage** | Cobertura de conceptos clave | BookSum |

---

## 💻 REQUISITOS

- **Python**: 3.8 o superior
- **PyTorch**: 2.0+
- **CUDA** (opcional): Para usar GPU
- **RAM**: 8 GB mínimo (16 GB recomendado)
- **Espacio**: 500 MB (solo proyecto) + dataset (se descarga automático)

---

## 🎓 SOBRE EL PROYECTO

- **Algoritmo**: MemSum (Memory-based Extractive Summarization)
- **Técnica**: Reinforcement Learning (REINFORCE)
- **Dataset**: BookSum (resúmenes de libros completos)
- **Papers**: 
  - BookSum: "BookSum: A Collection of Thousands of Book Summaries"
  - MemSum: "Neural Extractive Summarization with Side Information"

---

## 📝 COMANDOS IMPORTANTES

Ver archivo `COMANDOS.txt` para lista completa de comandos útiles.

---

## 🐛 SOLUCIÓN DE PROBLEMAS

**Error de CUDA/GPU**: El código funciona automáticamente en CPU.

**Error de memoria**: Reduce `batch_size` en el comando de entrenamiento.

**Error de dependencias**: `pip install --upgrade -r requirements.txt`

**Error de NLTK**: `python -c "import nltk; nltk.download('punkt')"`

---

## ✅ VERIFICACIÓN

Ejecuta para verificar que todo funciona:
```bash
python verificar_setup.py
```

---

## 📧 NOTAS

- El modelo ya está entrenado en `checkpoints/best_model.pt`
- El dataset BookSum se descarga automáticamente
- Funciona en CPU y GPU (detecta automáticamente)
- Compatible con Linux, macOS, Windows

---

¡Listo para generar resúmenes! 🎉
