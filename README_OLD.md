# MemSum para BookSum Dataset

Implementación de **MemSum** (Multi-step Episodic Markov decision process extractive SUMmarizer) adaptado para el dataset **BookSum** con soporte completo para GPU RTX 3050.

## 🚀 Descripción

Este proyecto implementa MemSum, un modelo de **resumen extractivo** que utiliza **aprendizaje por refuerzo** para seleccionar las oraciones más importantes de documentos largos. La implementación está optimizada para:

- 📚 **Dataset BookSum**: Resúmenes de capítulos de libros
- 🎮 **GPU RTX 3050**: Optimizado para 4GB de VRAM
- 🧠 **Memoria Episódica**: Evita redundancia recordando extracciones previas
- 🎯 **Aprendizaje por Refuerzo**: Política entrenada con recompensas ROUGE

## 🏗️ Arquitectura

### Componentes Principales:

1. **Sentence Encoder**: BiLSTM para codificar oraciones
2. **Document Encoder**: Transformer para contexto global
3. **Memory Module**: LSTM + Attention para historia de extracciones
4. **Extraction Policy**: Red neuronal para decisiones de extracción

### Flujo de Datos:
```
Texto → Oraciones → Sentence Encoder → Document Encoder
                                           ↓
Política ← Memory Module ← Historia de Extracciones
```

## 🛠️ Instalación

### Prerrequisitos
- Python 3.8+
- CUDA 13.0 (para GPU RTX 3050)
- 8GB+ RAM
- 20GB+ espacio libre

### 1. Clonar y configurar entorno
```bash
cd /home/lagusa/Documentos/TI3
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o .venv\Scripts\activate  # Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Verificar GPU
```bash
python -c "import torch; print(f'GPU disponible: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No disponible\"}')"
```

## 📊 Configuración

### Archivo de configuración principal: `configs/booksum_config.yaml`

Configuración optimizada para RTX 3050:
- **Batch size**: 4 (para 4GB VRAM)
- **Accumulation steps**: 8 (batch efectivo de 32)
- **Mixed precision**: Habilitado
- **Gradient checkpointing**: Para reducir memoria

## 🚂 Entrenamiento

### Entrenamiento básico
```bash
python train.py --config configs/booksum_config.yaml
```

### Entrenamiento con opciones avanzadas
```bash
python train.py \
    --config configs/booksum_config.yaml \
    --epochs 20 \
    --lr 1e-4 \
    --batch_size 4 \
    --wandb \
    --seed 42
```

### Reanudar entrenamiento
```bash
python train.py \
    --config configs/booksum_config.yaml \
    --resume checkpoints/checkpoint_epoch_10.pt
```

### Monitoreo durante entrenamiento
```bash
# En otra terminal
tail -f training.log

# O si usas wandb
# Ve a https://wandb.ai/tu-usuario/memsum-booksum
```

## 🧪 Evaluación

### Evaluar modelo entrenado
```bash
python evaluate.py checkpoints/best_model.pt \
    --config configs/booksum_config.yaml \
    --split test \
    --output results.json
```

### Generar resumen de texto específico
```bash
python evaluate.py checkpoints/best_model.pt \
    --text "Tu texto aquí para resumir..."
```

## 📁 Estructura del Proyecto

```
TI3/
├── src/
│   ├── config.py           # Configuración del modelo
│   ├── data_loader.py      # Carga y procesamiento de BookSum
│   ├── model.py           # Arquitectura MemSum
│   └── trainer.py         # Entrenamiento con RL
├── configs/
│   └── booksum_config.yaml # Configuración principal
├── scripts/
│   └── (scripts auxiliares)
├── data/                  # Datos y vocabulario
├── models/               # Modelos guardados
├── checkpoints/          # Checkpoints de entrenamiento
├── logs/                # Logs y métricas
├── train.py             # Script de entrenamiento
├── evaluate.py          # Script de evaluación
├── requirements.txt     # Dependencias
└── README.md           # Este archivo
```

## 🎯 Resultados Esperados

### Métricas ROUGE esperadas en BookSum:
- **ROUGE-1**: ~0.42-0.48
- **ROUGE-2**: ~0.18-0.24  
- **ROUGE-L**: ~0.38-0.45

### Tiempo de entrenamiento (RTX 3050):
- **Por época**: ~45-60 minutos
- **Entrenamiento completo**: ~12-15 horas

## 🔧 Optimizaciones para RTX 3050

### Configuraciones específicas:
```yaml
training:
  batch_size: 4              # Optimizado para 4GB VRAM
  accumulation_steps: 8      # Batch efectivo de 32
  
model:
  hidden_dim: 256           # Balanceado rendimiento/memoria
  max_doc_len: 500         # Longitud máxima de documento
  
device:
  mixed_precision: true     # Reduce uso de memoria 50%
  use_gpu: true
```

### Tips de optimización:
1. **Monitoring memoria**: `nvidia-smi` cada 30s
2. **Batch dinámico**: Reduce batch_size si hay OOM
3. **Gradient checkpointing**: Habilitado automáticamente
4. **Liberación caché**: `torch.cuda.empty_cache()` automático

## 🐛 Troubleshooting

### Error: CUDA Out of Memory
```bash
# Solución 1: Reducir batch size
python train.py --batch_size 2

# Solución 2: Sin mixed precision
# Editar config: mixed_precision: false

# Solución 3: Reducir dimensiones del modelo
# Editar config: hidden_dim: 128
```

### Error: Dataset no encontrado
```bash
# El código creará datos dummy automáticamente
# O descargar BookSum manualmente:
huggingface-cli login
python -c "from datasets import load_dataset; load_dataset('kmfoda/booksum')"
```

### Entrenamiento muy lento
```bash
# Verificar GPU
nvidia-smi

# Reducir num_workers si hay problemas I/O
python train.py --config configs/booksum_config.yaml
# Editar config: num_workers: 0
```

## 📚 Referencias

- **Paper Original**: [MemSum: Extractive Summarization of Long Documents Using Multi-Step Episodic Markov Decision Processes](https://aclanthology.org/2022.acl-long.450/)
- **BookSum Dataset**: [BookSum: A Collection of Datasets for Long-form Narrative Summarization](https://arxiv.org/abs/2105.08209)
- **Repositorio Original**: [nianlonggu/MemSum](https://github.com/nianlonggu/MemSum)

## 🤝 Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Añadir nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para detalles.

## 🙏 Agradecimientos

- Equipo original de MemSum (Gu et al., 2022)
- Dataset BookSum (Kryściński et al., 2021)  
- Comunidad de Transformers de Hugging Face
- PyTorch y NVIDIA por el soporte GPU

---

## 🚀 Quick Start

```bash
# 1. Activar entorno
source .venv/bin/activate

# 2. Entrenar modelo
python train.py --config configs/booksum_config.yaml --epochs 5

# 3. Evaluar
python evaluate.py checkpoints/best_model.pt

# 4. Generar resumen
python evaluate.py checkpoints/best_model.pt \
    --text "Tu texto largo aquí..."
```

**¡Listo para entrenar MemSum en BookSum con tu RTX 3050! 🚀**