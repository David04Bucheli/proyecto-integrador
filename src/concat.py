import itertools
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from . import config

MODALITY_ORDER = ("fotometria", "espectros", "metadatos")

DEFAULT_GRID = {
    "hidden_dim": [0, 128, 256],
    "dropout": [0.1, 0.3],
    "weight_decay": [0.0, 1e-4],
}


class Standardizer:

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float32)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def state(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}


class FusionMLP(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3,
                 num_classes: int = config.NUM_CLASSES):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def take_modalities(X: np.ndarray, names: Sequence[str], block_dim: int,
                    order: Sequence[str] = MODALITY_ORDER) -> np.ndarray:
    blocks = []
    for n in names:
        i = list(order).index(n)
        blocks.append(X[:, i * block_dim:(i + 1) * block_dim])
    return np.concatenate(blocks, axis=1)


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0,
                          labels=list(range(config.NUM_CLASSES))))


@torch.no_grad()
def _predict_logits(model: nn.Module, X: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def train_head(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
               hidden_dim: int = 128, dropout: float = 0.3, weight_decay: float = 0.0,
               lr: float = 1e-3, batch_size: int = 64, max_epochs: int = 100, patience: int = 10,
               seed: int = config.SEED, device: str = "cpu", verbose: bool = False) -> Dict:
    torch.manual_seed(seed)
    scaler = Standardizer().fit(X_train)
    Xtr = torch.as_tensor(scaler.transform(X_train), device=device)
    Xva = torch.as_tensor(scaler.transform(X_val), device=device)
    ytr = torch.as_tensor(np.asarray(y_train), dtype=torch.long, device=device)
    yva = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=device)

    model = FusionMLP(Xtr.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    history, best = [], {"f1": -1.0, "epoch": 0, "state": None}
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(Xtr), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        val_logits = _predict_logits(model, Xva)
        val_loss = criterion(val_logits, yva).item()
        val_pred = val_logits.argmax(1).cpu().numpy()
        val_f1 = _macro_f1(yva.cpu().numpy(), val_pred)
        val_acc = float((val_pred == yva.cpu().numpy()).mean())
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_loss": val_loss,
                        "val_acc": val_acc, "val_macro_f1": val_f1})
        if verbose:
            print(f"[{epoch:03d}] train_loss={total / len(Xtr):.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.3f} val_macroF1={val_f1:.3f}")
        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= patience:
            break

    model.load_state_dict(best["state"])
    model.eval()
    return {
        "model": model, "scaler": scaler, "history": history,
        "best_epoch": best["epoch"], "best_val_macro_f1": best["f1"],
        "train_seconds": time.time() - t0,
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "params": {"hidden_dim": hidden_dim, "dropout": dropout, "weight_decay": weight_decay,
                   "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience},
    }


def grid_search(X_train, y_train, X_val, y_val, grid: Optional[Dict[str, List]] = None,
                device: str = "cpu", seed: int = config.SEED, verbose: bool = True, **fixed) -> Dict:
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    rows, best = [], None
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        fit = train_head(X_train, y_train, X_val, y_val, device=device, seed=seed, **params, **fixed)
        row = {**params, "val_macro_f1": fit["best_val_macro_f1"], "best_epoch": fit["best_epoch"],
               "n_parameters": fit["n_parameters"], "seconds": round(fit["train_seconds"], 1)}
        rows.append(row)
        if verbose:
            print(row)
        better = best is None or fit["best_val_macro_f1"] > best["best_val_macro_f1"] + 1e-12
        tie = best is not None and abs(fit["best_val_macro_f1"] - best["best_val_macro_f1"]) <= 1e-12
        if better or (tie and fit["n_parameters"] < best["n_parameters"]):
            best = fit
    return {"best": best, "table": rows}


@torch.no_grad()
def predict_proba_head(fit: Dict, X: np.ndarray, y_true: Optional[np.ndarray] = None,
                       device: str = "cpu") -> Dict[str, np.ndarray]:
    model = fit["model"].to(device)
    Xt = torch.as_tensor(fit["scaler"].transform(X), device=device)
    probs = torch.softmax(_predict_logits(model, Xt), dim=1).cpu().numpy()
    out = {"probs": probs, "y_pred": probs.argmax(axis=1)}
    if y_true is not None:
        out["y_true"] = np.asarray(y_true)
    return out


def save_head(path, fit: Dict, extra: Optional[Dict] = None) -> None:
    torch.save({"state": fit["model"].state_dict(), "scaler": fit["scaler"].state(),
                "params": fit["params"], "best_epoch": fit["best_epoch"],
                "best_val_macro_f1": fit["best_val_macro_f1"], "history": fit["history"],
                "extra": extra or {}}, path)