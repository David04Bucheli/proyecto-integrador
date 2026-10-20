# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

from . import config


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


#para que el loop sea el mismo para las tres baselines
def train_one_epoch(model: nn.Module, dataloader, optimizer, device: str,
                     forward_fn: Callable) -> EpochResult:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad()
        logits, labels = forward_fn(model, batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return EpochResult(
        loss=total_loss / len(all_labels),
        accuracy=accuracy_score(all_labels, all_preds),
        macro_f1=f1_score(all_labels, all_preds, average="macro", zero_division=0),
    )


#exactitud, macro-F1, precisión/recall por clase y matriz de confusión
@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device: str, forward_fn: Callable,
             class_names: Optional[List[str]] = None) -> Dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        logits, labels = forward_fn(model, batch)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0,
        labels=list(range(config.NUM_CLASSES)),
    )
    names = class_names or [str(i) for i in range(config.NUM_CLASSES)]

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class": {
            names[i]: {"precision": precision[i], "recall": recall[i],
                       "f1": f1[i], "support": int(support[i])}
            for i in range(len(names))
        },
        "confusion_matrix": confusion_matrix(
            all_labels, all_preds, labels=list(range(config.NUM_CLASSES))),
        "y_true": all_labels,
        "y_pred": all_preds,
        "report_text": classification_report(
            all_labels, all_preds, labels=list(range(config.NUM_CLASSES)),
            target_names=names, zero_division=0),
    }

#entrena epochs y evalúa en validation tras cada una. Devuelve historial y métricas finales
def fit(model: nn.Module, train_loader, val_loader, forward_fn: Callable,
        device: str = "cpu", num_epochs: int = config.NUM_EPOCHS,
        lr: float = config.LEARNING_RATE, class_names: Optional[List[str]] = None,
        verbose: bool = True) -> Dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, num_epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, device, forward_fn)
        val_metrics = evaluate(model, val_loader, device, forward_fn, class_names)
        history.append({
            "epoch": epoch,
            "train_loss": train_result.loss, "train_acc": train_result.accuracy,
            "train_macro_f1": train_result.macro_f1,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })
        if verbose:
            print(f"[{epoch:02d}/{num_epochs}] "
                  f"train_loss={train_result.loss:.4f} train_acc={train_result.accuracy:.3f} | "
                  f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                  f"val_macroF1={val_metrics['macro_f1']:.3f}")

    return {"history": history, "final_val_metrics": val_metrics}


#forward_fn por modalidad

def forward_fn_photometry(model, batch):
    logits, _ = model(batch["photometry"], batch["photometry_lengths"])
    return logits, batch["label"]


def forward_fn_spectra(model, batch):
    logits, _ = model(batch["spectra"])
    return logits, batch["label"]


def forward_fn_metadata(model, batch):
    logits, _ = model(batch["metadata"])
    return logits, batch["label"]
