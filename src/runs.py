# -*- coding: utf-8 -*-
"""para entrenar con checkpoints, guardar todo lo que hace falta para votar/extraer embeddings sin reentrenar, y predecir con
probabilidades
"""
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import config, engine


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


#guarda el estado de la última época y el de la época con mejor macro-F1 en validation
def fit_with_checkpoints(model: nn.Module, train_loader, val_loader, forward_fn: Callable,
                         device: str = "cpu", num_epochs: int = config.NUM_EPOCHS,
                         lr: float = config.LEARNING_RATE,
                         class_names: Optional[List[str]] = None,
                         verbose: bool = True) -> Dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_f1, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        train_result = engine.train_one_epoch(model, train_loader, optimizer, device, forward_fn)
        val_metrics = engine.evaluate(model, val_loader, device, forward_fn, class_names)
        history.append({
            "epoch": epoch,
            "train_loss": train_result.loss, "train_acc": train_result.accuracy,
            "train_macro_f1": train_result.macro_f1,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"[{epoch:02d}/{num_epochs}] "
                  f"train_loss={train_result.loss:.4f} train_acc={train_result.accuracy:.3f} | "
                  f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                  f"val_macroF1={val_metrics['macro_f1']:.3f}")

    return {
        "history": history,
        "last_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "best_state": best_state,
        "best_epoch": best_epoch,
        "train_seconds": time.time() - t0,
        "n_parameters": count_parameters(model),
    }


def save_checkpoint(path, fit_result: Dict, extra: Optional[Dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "last_state": fit_result["last_state"],
        "best_state": fit_result["best_state"],
        "best_epoch": fit_result["best_epoch"],
        "history": fit_result["history"],
        "train_seconds": fit_result["train_seconds"],
        "n_parameters": fit_result["n_parameters"],
        "extra": extra or {},
    }, path)


def load_checkpoint_into(model: nn.Module, path, which: str = "last", device: str = "cpu") -> Dict:
    assert which in ("last", "best")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt[f"{which}_state"])
    model.to(device)
    model.eval()
    return ckpt


def save_history_json(path, history: List[Dict], extra: Optional[Dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "extra": extra or {}}, f, indent=2, ensure_ascii=False)


#predice un split completo y devuelve probabilidades, etiquetas y predicciones
@torch.no_grad()
def predict_proba(model: nn.Module, dataloader, device: str, forward_fn: Callable) -> Dict[str, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for batch in dataloader:
        batch = engine._move_batch_to_device(batch, device)
        logits, labels = forward_fn(model, batch)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return {"probs": probs, "y_true": labels, "y_pred": probs.argmax(axis=1)}


def save_predictions(path, outputs: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **outputs)


def load_predictions(path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}
