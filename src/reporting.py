# -*- coding: utf-8 -*-
"""Gráficos y tablas para el Entregable 2. Solo grafican lo que se les pasa."""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

from . import config


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = config.CLASS_NAMES,
                          title: str = "", normalize: bool = True, path: Optional[str] = None):
    """Matriz 10x10. normalize=True muestra el recall por fila (cada fila suma 1)."""
    plt = _plt()
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    thr = shown.max() / 2
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_history(histories: Dict[str, List[Dict]], path: Optional[str] = None):
    """Curvas por época (loss y macro-F1 de validación) de varios modelos."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], label=f"{name} train")
        axes[0].plot(ep, [r["val_loss"] for r in h], "--", label=f"{name} val")
        axes[1].plot(ep, [r["val_macro_f1"] for r in h], label=name)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Época"); axes[0].legend(fontsize=7)
    axes[1].set_title("Macro-F1 en validación"); axes[1].set_xlabel("Época"); axes[1].legend()
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def per_class_table(metrics_by_name: Dict[str, Dict], metric: str = "f1") -> pd.DataFrame:
    """Filas = clases, columnas = modelos/estrategias; valor = f1 (o precision/recall) por clase."""
    cols = {name: {c: v[metric] for c, v in m["per_class"].items()} for name, m in metrics_by_name.items()}
    return pd.DataFrame(cols)


def summary_table(metrics_by_name: Dict[str, Dict]) -> pd.DataFrame:
    return pd.DataFrame({name: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                         for name, m in metrics_by_name.items()}).T.round(4)
