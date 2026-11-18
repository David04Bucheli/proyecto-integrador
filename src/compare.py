import re
import unicodedata
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import config

LABELS = list(range(config.NUM_CLASSES))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     n_boot: int = 2000, seed: int = config.SEED) -> Dict[str, float]:
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas_f1 = np.empty(n_boot)
    deltas_acc = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas_f1[b] = macro_f1(y_true[idx], pred_a[idx]) - macro_f1(y_true[idx], pred_b[idx])
        deltas_acc[b] = (pred_a[idx] == y_true[idx]).mean() - (pred_b[idx] == y_true[idx]).mean()
    return {
        "delta_macro_f1": macro_f1(y_true, pred_a) - macro_f1(y_true, pred_b),
        "ci95_macro_f1": (float(np.percentile(deltas_f1, 2.5)), float(np.percentile(deltas_f1, 97.5))),
        "p_a_better_macro_f1": float((deltas_f1 > 0).mean()),
        "delta_accuracy": float((pred_a == y_true).mean() - (pred_b == y_true).mean()),
        "ci95_accuracy": (float(np.percentile(deltas_acc, 2.5)), float(np.percentile(deltas_acc, 97.5))),
    }


def bootstrap_table(y_true: np.ndarray, preds: Dict[str, np.ndarray], reference: str,
                    n_boot: int = 2000, seed: int = config.SEED) -> pd.DataFrame:
    rows = {}
    for name, pred in preds.items():
        if name == reference:
            continue
        r = paired_bootstrap(y_true, pred, preds[reference], n_boot=n_boot, seed=seed)
        rows[name] = {
            "Δ macro-F1": round(r["delta_macro_f1"], 4),
            "IC95 inf": round(r["ci95_macro_f1"][0], 4),
            "IC95 sup": round(r["ci95_macro_f1"][1], 4),
            "P(mejor)": round(r["p_a_better_macro_f1"], 3),
            "Δ exactitud": round(r["delta_accuracy"], 4),
        }
    return pd.DataFrame(rows).T


def results_table(entries: Dict[str, Dict]) -> pd.DataFrame:
    rows = {}
    for name, e in entries.items():
        row = {}
        for split in ("validation", "test"):
            if split in e:
                y_true, y_pred = e[split]["y_true"], e[split]["y_pred"]
                row[f"exactitud ({split})"] = accuracy_score(y_true, y_pred)
                row[f"macro-F1 ({split})"] = macro_f1(y_true, y_pred)
        for k in ("parametros_entrenables", "segundos_entrenamiento"):
            if k in e:
                row[k] = e[k]
        rows[name] = row
    return pd.DataFrame(rows).T


def plot_strategy_bars(table: pd.DataFrame, column: str = "macro-F1 (test)", path: Optional[str] = None,
                       title: str = ""):
    import matplotlib.pyplot as plt
    data = table[column].sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.2))
    ax.barh(data.index, data.values, color="#4C72B0")
    for i, v in enumerate(data.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    lo = max(0.0, float(data.min()) - 0.1)
    ax.set_xlim(lo, min(1.0, float(data.max()) + 0.05))
    ax.set_xlabel(column)
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")