# -*- coding: utf-8 -*-
"""
Prueba de humo (smoke test): NO usa el dataset real (este sandbox no
tiene acceso a huggingface.co). Genera datos sintéticos con las mismas
formas documentadas del dataset real para verificar que el pipeline
completo -collate, los tres modelos, el loop de entrenamiento y la
extracción de embeddings- corre sin errores antes de que el estudiante
lo use contra los datos reales en Colab.

Ejecutar desde la raíz del repositorio:
    python tests/test_smoke.py
"""
import sys
from pathlib import Path

# Permite "python tests/test_smoke.py" desde cualquier directorio: añade la
# raíz del repositorio (un nivel arriba de tests/) al sys.path para que
# "from src import ..." funcione sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config, engine, embeddings as emb_module
from src.datasets import collate_multimodal
from src.models import PhotometryGRUEncoder, SpectraCNNEncoder, MetadataMLPEncoder, UnimodalClassifier
from src.audit import _fingerprint_split

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

N_TRAIN, N_VAL = 64, 16
# Confirmado contra el dataset real en Colab (ver README): metadata tiene
# 34 valores por objeto y NO existe columna de ID (de ahí la auditoría por
# huella digital en src/audit.py, probada en la sección 6 más abajo).
METADATA_DIM = 34


def make_fake_split(n):
    """Simula lo que devolvería AstroMultimodalDataset.__getitem__ para
    n objetos, con longitudes de curva de luz variables (10-60 puntos)."""
    items = []
    for _ in range(n):
        T = np.random.randint(10, 60)
        items.append({
            "photometry": torch.randn(T, 9),
            "spectra": torch.randn(3, 2575),
            "metadata": torch.randn(METADATA_DIM),
            "label": int(np.random.randint(0, config.NUM_CLASSES)),
        })
    return items


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_loader(n, batch_size=8):
    ds = FakeDataset(make_fake_split(n))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal)


def check(name, condition):
    status = "OK" if condition else "FALLÓ"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


print("=== 1) Collate + formas de batch ===")
train_loader = make_loader(N_TRAIN)
val_loader = make_loader(N_VAL)
batch = next(iter(train_loader))
check("photometry es (B, T_max, 9)", batch["photometry"].ndim == 3 and batch["photometry"].shape[2] == 9)
check("photometry_lengths tiene tamaño B", batch["photometry_lengths"].shape[0] == batch["photometry"].shape[0])
check("spectra es (B, 3, 2575)", tuple(batch["spectra"].shape[1:]) == (3, 2575))
check("metadata es (B, METADATA_DIM)", batch["metadata"].shape[1] == METADATA_DIM)
check("label es (B,)", batch["label"].ndim == 1)

print("\n=== 2) Forward pass de cada codificador ===")
device = "cpu"

gru_model = UnimodalClassifier(PhotometryGRUEncoder(input_dim=9), num_classes=config.NUM_CLASSES).to(device)
logits, emb = gru_model(batch["photometry"].to(device), batch["photometry_lengths"].to(device))
check("GRU logits shape (B, NUM_CLASSES)", tuple(logits.shape) == (batch["label"].shape[0], config.NUM_CLASSES))
check("GRU embedding shape (B, EMBEDDING_DIM)", tuple(emb.shape) == (batch["label"].shape[0], config.EMBEDDING_DIM))

cnn_model = UnimodalClassifier(SpectraCNNEncoder(in_channels=3), num_classes=config.NUM_CLASSES).to(device)
logits, emb = cnn_model(batch["spectra"].to(device))
check("CNN logits shape (B, NUM_CLASSES)", tuple(logits.shape) == (batch["label"].shape[0], config.NUM_CLASSES))
check("CNN embedding shape (B, EMBEDDING_DIM)", tuple(emb.shape) == (batch["label"].shape[0], config.EMBEDDING_DIM))

mlp_model = UnimodalClassifier(MetadataMLPEncoder(input_dim=METADATA_DIM), num_classes=config.NUM_CLASSES).to(device)
logits, emb = mlp_model(batch["metadata"].to(device))
check("MLP logits shape (B, NUM_CLASSES)", tuple(logits.shape) == (batch["label"].shape[0], config.NUM_CLASSES))
check("MLP embedding shape (B, EMBEDDING_DIM)", tuple(emb.shape) == (batch["label"].shape[0], config.EMBEDDING_DIM))

print("\n=== 3) Un par de épocas de entrenamiento real (2 épocas, datos sintéticos) ===")
result = engine.fit(
    gru_model, train_loader, val_loader,
    forward_fn=engine.forward_fn_photometry, device=device, num_epochs=2,
    class_names=config.CLASS_NAMES, verbose=True,
)
check("fit() devuelve historial de 2 épocas", len(result["history"]) == 2)
check("fit() devuelve métricas finales de validación", "accuracy" in result["final_val_metrics"])

print("\n=== 4) Evaluación completa (matriz de confusión, reporte por clase) ===")
val_metrics = engine.evaluate(gru_model, val_loader, device, engine.forward_fn_photometry, config.CLASS_NAMES)
check("matriz de confusión NUM_CLASSES x NUM_CLASSES",
      val_metrics["confusion_matrix"].shape == (config.NUM_CLASSES, config.NUM_CLASSES))
check("hay macro_f1 numérico", isinstance(val_metrics["macro_f1"], float))

print("\n=== 5) Extracción y concatenación de embeddings ===")
gru_out = emb_module.extract_embeddings(gru_model, val_loader, device, engine.forward_fn_photometry)
cnn_out = emb_module.extract_embeddings(cnn_model, val_loader, device, engine.forward_fn_spectra)
mlp_out = emb_module.extract_embeddings(mlp_model, val_loader, device, engine.forward_fn_metadata)

check("embeddings GRU shape (N_VAL, EMBEDDING_DIM)",
      gru_out["embeddings"].shape == (N_VAL, config.EMBEDDING_DIM))

# OJO: aquí concatenamos con distinto orden de batches por diseño de la
# prueba (val_loader usa shuffle=True); en el pipeline real hay que usar
# shuffle=False al extraer embeddings para que las tres modalidades
# queden alineadas por índice. Lo dejamos como comentario-recordatorio.
fused = emb_module.concatenate_modalities(
    gru_out["embeddings"], cnn_out["embeddings"], mlp_out["embeddings"], gru_out["labels"])
check("embedding fusionado shape (N_VAL, 3*EMBEDDING_DIM)",
      fused["embeddings"].shape == (N_VAL, 3 * config.EMBEDDING_DIM))

print("\n=== 6) Auditoría de fuga por huella digital de metadatos (lógica, sin red) ===")
# AstroM3Processed no tiene columna de ID (confirmado en Colab), así que la
# auditoría real (src/audit.py) identifica objetos repetidos por su vector
# de metadatos. Esto NO reemplaza la auditoría real contra el dataset real:
# solo confirma, con datos sintéticos, que la lógica de intersección de
# huellas digitales detecta correctamente tanto la ausencia de fuga como
# una fuga inyectada a propósito.
fake_train_metadata = [[i + j * 0.001 for j in range(34)] for i in range(50)]
fake_test_metadata_clean = [[1000 + i + j * 0.001 for j in range(34)] for i in range(20)]
fake_test_metadata_leaky = fake_test_metadata_clean[:15] + fake_train_metadata[:5]

train_fps = set(_fingerprint_split({"metadata": fake_train_metadata}))
test_fps_clean = set(_fingerprint_split({"metadata": fake_test_metadata_clean}))
test_fps_leaky = set(_fingerprint_split({"metadata": fake_test_metadata_leaky}))

check("sin fuga inyectada -> intersección vacía", len(train_fps & test_fps_clean) == 0)
check("con 5 filas de fuga inyectada -> intersección de tamaño 5", len(train_fps & test_fps_leaky) == 5)

print("\n✅ TODAS LAS PRUEBAS DE HUMO PASARON (con datos sintéticos).")
print("   Falta validar contra el dataset real en Colab (requiere internet):")
print("   1) run_self_check()      -> valida el método con un subset pequeño (sub10)")
print("   2) run_leakage_audit()   -> la auditoría real (full_42 vs sub50_42)")
