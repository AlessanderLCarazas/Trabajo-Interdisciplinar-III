# 🚀 SETUP RÁPIDO - PROYECTO MEMSUM

## 📦 INSTALACIÓN

### 1. Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar Python environment
El proyecto funciona tanto en **CPU como GPU** (CUDA).
PyTorch detectará automáticamente si hay GPU disponible.

---

## 🎯 ENTRENAR MODELO

### Entrenamiento corto (5 epochs - prueba)
```bash
python train.py --config configs/booksum_config.yaml --epochs 5 --batch_size 2
```

### Entrenamiento completo (40 epochs)
```bash
python train.py --config configs/booksum_config.yaml --epochs 40 --batch_size 2
```

**Nota**: El batch_size puede ajustarse según memoria disponible.
- GPU: batch_size 4-8
- CPU: batch_size 1-2

---

## 🔍 EVALUAR MODELO

```bash
python evaluate.py --model_path checkpoints/best_model.pt
```

---

## 🌐 LEVANTAR INTERFAZ WEB

### Interfaz básica
```bash
python app.py
```

### Interfaz avanzada (con BookSum y todas las métricas)
```bash
python app_advanced.py
```

Acceder en: **http://localhost:8000**

---

## 📄 RESUMIR PDFs/EPUBs

### PDF
```bash
python scripts/summarize_pdf.py --pdf_path "ruta/al/archivo.pdf" --output "resumen.txt"
```

### EPUB
```bash
python scripts/summarize_epub.py --epub_path "ruta/al/archivo.epub" --output "resumen.txt"
```

---

## 📂 ESTRUCTURA DEL PROYECTO

```
jalar/
├── train.py              # Script de entrenamiento
├── evaluate.py           # Script de evaluación
├── app.py                # Interfaz web básica
├── app_advanced.py       # Interfaz web completa
├── requirements.txt      # Dependencias
├── quick_start.sh        # Script de inicio rápido
├── src/                  # Código fuente
│   ├── __init__.py
│   ├── model.py          # Arquitectura MemSum
│   ├── trainer.py        # Lógica de entrenamiento
│   ├── data_loader.py    # Carga de datos BookSum
│   ├── config.py         # Configuración
│   └── fusion.py         # Fusion layers
├── configs/              # Archivos de configuración
│   ├── booksum_config.yaml
│   └── booksum_full_config.yaml
├── checkpoints/          # Modelos entrenados
│   └── best_model.pt     # Mejor modelo (14MB)
├── data/                 # Datos procesados
│   └── vocab.pkl         # Vocabulario
└── scripts/              # Scripts auxiliares
    ├── summarize_pdf.py
    └── summarize_epub.py
```

---

## ⚙️ CONFIGURACIÓN

Editar `configs/booksum_config.yaml` para ajustar:
- Número de epochs
- Batch size
- Learning rate
- Tamaño del modelo
- Parámetros de RL

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### Error de CUDA/GPU
Si no tienes GPU, el código funciona automáticamente en CPU.

### Error de memoria
Reduce el `batch_size` en el comando de entrenamiento.

### Error de dependencias
Ejecuta: `pip install --upgrade -r requirements.txt`

---

## 📊 MÉTRICAS EVALUADAS

- **ROUGE-1, 2, L**: Coincidencia léxica
- **BERTScore**: Similitud semántica
- **Content Coverage**: Cobertura de conceptos clave

---

## ✅ VERIFICACIÓN RÁPIDA

```bash
# Probar que todo funciona
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())"
```

¡Listo para entrenar y resumir! 🎉
