#!/usr/bin/env python3
"""
Visualización de la arquitectura MemSum usando torchview.
Genera un gráfico en formato imagen (PNG/SVG/PDF) del grafo computacional a nivel de módulos/capas.

Uso:
  python scripts/visualize_architecture.py --out models/memsum_architecture.png --format png

Notas:
- Requiere instalar: torchview y graphviz (paquete Python). En Linux también es necesario el binario de graphviz (sudo apt-get install graphviz).
- Usa un input dummy pequeño para no requerir GPU.
"""

import os
import sys
import argparse

# Asegurar que src/ esté en el path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Si se ejecuta desde el repo, ROOT es el padre de scripts/
ROOT = os.path.dirname(SCRIPT_DIR)
# Fallback: si no existe src/ en ROOT, probar CWD/src
SRC1 = os.path.join(ROOT, 'src')
SRC2 = os.path.join(os.getcwd(), 'src')
for cand in (SRC1, SRC2):
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.append(cand)

import torch

try:
    from src.config import Config
    from src.model import MemSum
except ModuleNotFoundError:
    # Fallback robusto: cargar módulos por ruta
    import importlib.util
    from importlib.machinery import SourceFileLoader
    config_path = os.path.join(ROOT, 'src', 'config.py')
    model_path = os.path.join(ROOT, 'src', 'model.py')
    if not (os.path.isfile(config_path) and os.path.isfile(model_path)):
        raise
    Config = SourceFileLoader('memsum_config', config_path).load_module().Config
    MemSum = SourceFileLoader('memsum_model', model_path).load_module().MemSum


def get_args():
    parser = argparse.ArgumentParser(description='Visualizar arquitectura MemSum')
    parser.add_argument('--out', type=str, default=os.path.join(ROOT, 'models', 'memsum_architecture'),
                        help='Ruta de salida sin extensión (se añade .png/.svg/.pdf según formato)')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'svg', 'pdf'],
                        help='Formato de salida de la imagen')
    parser.add_argument('--doc_len', type=int, default=32, help='Número de oraciones dummy')
    parser.add_argument('--sent_len', type=int, default=40, help='Longitud de oración dummy')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Tamaño de vocabulario dummy')
    return parser.parse_args()


def ensure_dirs(path_without_ext: str):
    dirpath = os.path.dirname(path_without_ext)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def build_dummy_inputs(batch_size: int, doc_len: int, sent_len: int, vocab_size: int, device: torch.device):
    sentences = torch.randint(low=1, high=vocab_size, size=(batch_size, doc_len, sent_len), dtype=torch.long, device=device)
    mask = torch.ones((batch_size, doc_len), dtype=torch.float, device=device)
    return sentences, mask


def save_model_summary(model: torch.nn.Module, out_txt_path: str):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(out_txt_path, 'w') as f:
        f.write('MemSum Architecture Summary\n')
        f.write('===========================\n\n')
        f.write(str(model))
        f.write('\n\n')
        f.write(f'Total parameters: {total_params:,}\n')
        f.write(f'Trainable parameters: {trainable_params:,}\n')


def main():
    args = get_args()
    device = torch.device('cpu')  # Forzar CPU para trazado estable

    # Cargar configuración (usa defaults si no hay YAML)
    config_path = os.path.join(ROOT, 'configs', 'booksum_config.yaml')
    config = Config(config_path if os.path.exists(config_path) else None)

    # Construir modelo
    model = MemSum(vocab_size=args.vocab_size, config=config)
    model.eval()
    model.to(device)

    # Entradas dummy
    sentences, mask = build_dummy_inputs(batch_size=1, doc_len=args.doc_len, sent_len=args.sent_len,
                                         vocab_size=args.vocab_size, device=device)

    # Crear gráfico con torchview
    try:
        from torchview import draw_graph
    except ImportError as e:
        print('[ERROR] torchview no está instalado. Instala con: pip install torchview graphviz')
        raise e

    # Generar grafo; roll=True intenta compactar módulos repetidos
    graph = draw_graph(model,
                       input_data=(sentences, mask),
                       graph_name='MemSum',
                       roll=True,
                       depth=4)  # profundidad moderada para legibilidad

    # Preparar rutas de salida
    ensure_dirs(args.out)
    out_path_no_ext = args.out if not args.out.endswith(f'.{args.format}') else args.out[:-len(f'.{args.format}')]

    # Guardar DOT (fuente del grafo) y resumen textual SIEMPRE
    dot_path = out_path_no_ext + '.dot'
    with open(dot_path, 'w') as f:
        f.write(graph.visual_graph.source)

    summary_txt = out_path_no_ext + '_summary.txt'
    save_model_summary(model, summary_txt)

    # Intentar renderizar imagen (requiere binario 'dot')
    final_path = f"{out_path_no_ext}.{args.format}"
    try:
        # render() espera path sin extensión
        graph.visual_graph.render(out_path_no_ext, format=args.format, cleanup=True)
        print(f"[OK] Arquitectura guardada en: {final_path}")
    except Exception as e:
        print("[WARN] No se pudo renderizar la imagen. Probablemente falta el binario 'dot' de Graphviz.")
        print("       Sugerencia (Debian/Ubuntu): sudo apt-get update && sudo apt-get install -y graphviz")
        print(f"       Se guardó el archivo DOT en: {dot_path}")
        print(f"       Puedes abrir el .dot en https://dreampuf.github.io/GraphvizOnline/ o renderizarlo luego.")

    print(f"[OK] Resumen del modelo: {summary_txt}")


if __name__ == '__main__':
    main()
