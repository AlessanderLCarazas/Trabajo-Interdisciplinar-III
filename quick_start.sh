#!/bin/bash
# Quick Start Script for MemSum Training
# Autor: GitHub Copilot
# Descripción: Script para iniciar entrenamiento de MemSum rápidamente
set -e  # Exit on error
# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
# Función para imprimir mensajes con colores
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}
# Verificar si estamos en el directorio correcto
if [[ ! -f "train.py" || ! -d "src" ]]; then
    print_error "Por favor ejecuta este script desde el directorio TI3"
    exit 1
fi
print_header "MEMSUM QUICK START"
# Verificar entorno virtual
if [[ ! -d ".venv" ]]; then
    print_error "Entorno virtual no encontrado. Por favor ejecuta primero la configuración."
    exit 1
fi
print_status "Activando entorno virtual..."
source .venv/bin/activate
# Verificar GPU
print_status "Verificando GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f' GPU disponible: {torch.cuda.get_device_name(0)}')
    print(f' Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print(' GPU no disponible, usando CPU')
"
# Menú de opciones
print_header "Selecciona una opción:"
echo "1. Ejecutar tests de verificación"
echo "2. Test de entrenamiento rápido (datos dummy)"
echo "3. Entrenamiento rápido BookSum (1 época)"
echo "4. Entrenamiento completo (15 épocas)"
echo "5. Entrenar con monitoreo Wandb"
echo "6. Reanudar entrenamiento desde checkpoint"
echo "7. Evaluar modelo existente"
echo "8. Generar resumen de texto"
echo "9. Ver configuración actual"
read -p "Ingresa tu opción (1-9): " choice
case $choice in
    1)
        print_header " EJECUTANDO TESTS"
        python simple_test.py

        if [[ $? -eq 0 ]]; then
            print_status " Todos los tests pasaron! Sistema listo para entrenar."
        else
            print_error "* Algunos tests fallaron. Revisa la configuración."
        fi
        ;;
    2)
        print_header " TEST DE ENTRENAMIENTO RAPIDO"
        print_status "Probando pipeline completo con datos dummy (2-3 minutos)..."
        python fast_train_test.py

        if [[ $? -eq 0 ]]; then
            print_status " Test de entrenamiento completado exitosamente!"
        else
            print_error " Error en test de entrenamiento"
        fi
        ;;
    3)
        print_header " ENTRENAMIENTO RAPIDO BOOKSUM"
        print_status "Iniciando entrenamiento con BookSum (1 época, ~15-20 min)..."
        print_warning "Esto descargará y procesará BookSum dataset..."
        python train.py \
            --config configs/booksum_config.yaml \
            --epochs 1 \
            --batch_size 2

        if [[ $? -eq 0 ]]; then
            print_status "Entrenamiento BookSum completado exitosamente!"
        else
            print_error "Error en entrenamiento BookSum"
        fi
        ;;
    4)
        print_header "ENTRENAMIENTO COMPLETO"
        print_warning "Esto tomará 12-15 horas. ¿Continuar? (y/n)"
        read -p "" confirm
        if [[ $confirm == [yY] ]]; then
            print_status "Iniciando entrenamiento completo..."
            python train.py --config configs/booksum_config.yaml
        else
            print_status "Entrenamiento cancelado."
        fi
        ;;
    5)
        print_header " ENTRENAMIENTO CON WANDB"
        print_status "Iniciando entrenamiento con monitoreo Wandb..."
        python train.py \
            --config configs/booksum_config.yaml \
            --wandb
        ;;
    6)
        print_header " REANUDAR ENTRENAMIENTO"
        if [[ ! -d "checkpoints" ]] || [[ -z "$(ls -A checkpoints 2>/dev/null)" ]]; then
            print_error "No se encontraron checkpoints."
            exit 1
        fi

        echo "Checkpoints disponibles:"
        ls -la checkpoints/*.pt 2>/dev/null || echo "No hay checkpoints .pt"
        read -p "Ingresa el nombre del checkpoint (ej: checkpoint_epoch_5.pt): " checkpoint

        if [[ -f "checkpoints/$checkpoint" ]]; then
            python train.py \
                --config configs/booksum_config.yaml \
                --resume "checkpoints/$checkpoint"
        else
            print_error "Checkpoint no encontrado: checkpoints/$checkpoint"
        fi
        ;;
    7)
        print_header " EVALUAR MODELO"
        if [[ ! -f "checkpoints/best_model.pt" ]]; then
            print_error "No se encontró best_model.pt. Entrena un modelo primero."
            exit 1
        fi

        print_status "Evaluando mejor modelo..."
        python evaluate.py checkpoints/best_model.pt \
            --config configs/booksum_config.yaml \
            --output evaluation_results.json
        ;;
    8)
        print_header " GENERAR RESUMEN"
        if [[ ! -f "checkpoints/best_model.pt" ]]; then
            print_error "No se encontró best_model.pt. Entrena un modelo primero."
            exit 1
        fi

        echo "Ingresa el texto a resumir (termina con una línea vacía):"
        text=""
        while IFS= read -r line; do
            [[ -z "$line" ]] && break
            text="$text $line"
        done

        if [[ -n "$text" ]]; then
            python evaluate.py checkpoints/best_model.pt --text "$text"
        else
            print_error "No se ingresó texto."
        fi
        ;;
    9)
        print_header " CONFIGURACION ACTUAL"
        echo "Configuración de entrenamiento:"
        cat configs/booksum_config.yaml
        ;;
    *)
        print_error "Opción no válida. Por favor selecciona 1-9."
        exit 1
        ;;
esac
print_status " Operacion completada!"
# Información adicional
print_header " INFORMACION UTIL"
echo "• Logs de entrenamiento: tail -f training.log"
echo "• Monitorear GPU: watch nvidia-smi"
echo "• Checkpoints: ls -la checkpoints/"
echo "• Resultados: ls -la logs/"
echo "• README completo: cat README.md"
