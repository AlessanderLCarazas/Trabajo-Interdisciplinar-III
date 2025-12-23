#!/usr/bin/env python
"""
Script de verificación para asegurar que todo está listo
para entrenar y ejecutar MemSum en la nueva laptop.
"""

import sys
import os
from pathlib import Path

def check_files():
    """Verifica que todos los archivos esenciales existan"""
    print("🔍 Verificando archivos esenciales...")
    
    required_files = [
        "train.py",
        "evaluate.py",
        "app.py",
        "app_advanced.py",
        "requirements.txt",
        "src/__init__.py",
        "src/model.py",
        "src/trainer.py",
        "src/data_loader.py",
        "src/config.py",
        "src/fusion.py",
        "configs/booksum_config.yaml",
        "configs/booksum_full_config.yaml",
        "checkpoints/best_model.pt",
        "data/vocab.pkl",
        "scripts/summarize_pdf.py",
        "scripts/summarize_epub.py",
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            print(f"  ❌ Falta: {file}")
        else:
            print(f"  ✅ {file}")
    
    if missing:
        print(f"\n⚠️  Faltan {len(missing)} archivos")
        return False
    else:
        print("\n✅ Todos los archivos presentes")
        return True

def check_python_version():
    """Verifica la versión de Python"""
    print("\n🐍 Verificando Python...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("  ✅ Versión compatible")
        return True
    else:
        print("  ⚠️  Se recomienda Python 3.8+")
        return False

def check_imports():
    """Verifica que las dependencias principales se puedan importar"""
    print("\n📦 Verificando dependencias...")
    
    dependencies = {
        "torch": "PyTorch",
        "datasets": "HuggingFace Datasets",
        "transformers": "HuggingFace Transformers",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "rouge_score": "ROUGE Score",
        "bert_score": "BERTScore",
        "pdfminer": "PDFMiner",
        "nltk": "NLTK",
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            missing.append(name)
            print(f"  ❌ {name} - NO INSTALADO")
    
    if missing:
        print(f"\n⚠️  Faltan {len(missing)} dependencias")
        print("Ejecuta: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Todas las dependencias instaladas")
        return True

def check_cuda():
    """Verifica disponibilidad de CUDA/GPU"""
    print("\n🎮 Verificando GPU/CUDA...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"  ✅ GPU disponible: {device_name}")
            print(f"  📊 GPUs detectadas: {device_count}")
            print(f"  💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("  ℹ️  GPU no disponible - se usará CPU")
            print("  (Funcionará pero será más lento)")
        
        return True
    except Exception as e:
        print(f"  ⚠️  Error verificando CUDA: {e}")
        return False

def check_model():
    """Verifica que el modelo cargue correctamente"""
    print("\n🤖 Verificando modelo entrenado...")
    
    try:
        import torch
        model_path = Path("checkpoints/best_model.pt")
        
        if not model_path.exists():
            print("  ❌ Modelo no encontrado")
            return False
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  📦 Tamaño: {size_mb:.2f} MB")
        
        # Intentar cargar el modelo
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"  ✅ Modelo cargado correctamente")
        
        if 'epoch' in checkpoint:
            print(f"  📊 Epoch: {checkpoint['epoch']}")
        
        return True
    except Exception as e:
        print(f"  ⚠️  Error cargando modelo: {e}")
        return False

def main():
    print("=" * 70)
    print("  VERIFICACIÓN DE SETUP - PROYECTO MEMSUM")
    print("=" * 70)
    
    results = []
    
    results.append(("Archivos", check_files()))
    results.append(("Python", check_python_version()))
    results.append(("Dependencias", check_imports()))
    results.append(("GPU/CUDA", check_cuda()))
    results.append(("Modelo", check_model()))
    
    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    
    for name, status in results:
        icon = "✅" if status else "⚠️"
        print(f"{icon} {name}: {'OK' if status else 'REVISAR'}")
    
    all_ok = all(status for _, status in results)
    
    print("\n" + "=" * 70)
    if all_ok:
        print("🎉 ¡TODO LISTO! Puedes comenzar a entrenar y usar el modelo.")
        print("\nPróximos pasos:")
        print("  1. Entrenar: python train.py --config configs/booksum_config.yaml")
        print("  2. Evaluar: python evaluate.py")
        print("  3. Interfaz: python app_advanced.py")
    else:
        print("⚠️  Hay algunos problemas que resolver.")
        print("Revisa las secciones marcadas arriba.")
        print("\nPara instalar dependencias:")
        print("  pip install -r requirements.txt")
    print("=" * 70)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
