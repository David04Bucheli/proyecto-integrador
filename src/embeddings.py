# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from . import engine

#corre el codificador sobre todo un split y devuelve embeddings + etiquetas como arrays
@torch.no_grad()
def extract_embeddings(model: nn.Module, dataloader, device: str,
                        forward_fn) -> Dict[str, np.ndarray]:
    model.eval()
    all_embeddings, all_labels = [], []

    for batch in dataloader:
        batch = engine._move_batch_to_device(batch, device)
        _, labels = forward_fn_labels_only(batch)
        embedding = _forward_embedding_only(model, batch, forward_fn)
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return {
        "embeddings": np.concatenate(all_embeddings, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
    }


def forward_fn_labels_only(batch):
    return None, batch["label"]


#reutiliza la lógica de forward_fn pero para los embeddings
def _forward_embedding_only(model, batch, forward_fn):
    if forward_fn is engine.forward_fn_photometry:
        _, embedding = model(batch["photometry"], batch["photometry_lengths"])
    elif forward_fn is engine.forward_fn_spectra:
        _, embedding = model(batch["spectra"])
    elif forward_fn is engine.forward_fn_metadata:
        _, embedding = model(batch["metadata"])
    else:
        raise ValueError("forward_fn no reconocido")
    return embedding


def save_embeddings(path: str, embeddings: np.ndarray, labels: np.ndarray) -> None:
    np.savez(path, embeddings=embeddings, labels=labels)
    print(f"Guardado: {path}  (embeddings: {embeddings.shape}, labels: {labels.shape})")


def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {"embeddings": data["embeddings"], "labels": data["labels"]}


#concatena los tres embeddings
def concatenate_modalities(photometry_emb: np.ndarray, spectra_emb: np.ndarray,
                            metadata_emb: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    assert photometry_emb.shape[0] == spectra_emb.shape[0] == metadata_emb.shape[0], \
        "Las tres modalidades deben tener el mismo número de ejemplos y en el mismo orden."
    fused = np.concatenate([photometry_emb, spectra_emb, metadata_emb], axis=1)
    return {"embeddings": fused, "labels": labels}
