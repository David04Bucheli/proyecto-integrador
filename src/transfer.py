import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import config, engine

ASTROM3_REPO = "https://github.com/MeriDK/AstroM3.git"
ASTROM3_COMMIT = "7f296c22729e87b92d98c1a1868699be73c2722e"
ASTROM3_DIR = Path("external") / "AstroM3"
SEQ_LEN = 200

FINETUNE_HP = {
    "fusion": "avg",
    "p_dropout": 0.08116365514804472,
    "s_dropout": 0.03454530839892027,
    "m_dropout": 0.2721805507275124,
    "lr": 0.0038789663081484622,
    "beta1": 0.8628516428717157,
    "beta2": 0.999,
    "weight_decay": 8.297980301069488e-05,
    "epochs": 50,
    "batch_size": 512,
    "warmup_epochs": 10,
    "scheduler_factor": 0.2571022327438819,
    "scheduler_patience": 3,
    "early_stopping_patience": 6,
}


def fetch_astrom3_code(root: Path = ASTROM3_DIR, commit: str = ASTROM3_COMMIT) -> Path:
    root = Path(root)
    if not (root / "src" / "model.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", ASTROM3_REPO, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "-q", commit], check=True)
    return root


def import_astrom3(root: Path = ASTROM3_DIR):
    src = Path(root) / "src"
    for name, filename in (("informer", "informer.py"), ("astrom3_model", "model.py")):
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, src / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["astrom3_model"]


def load_pretrained_files(source: str = config.CLIP_MODEL_NAME):
    from safetensors.torch import load_file
    if os.path.isdir(source):
        cfg_path, weights_path = Path(source) / "config.json", Path(source) / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_id=source, filename="config.json")
        weights_path = hf_hub_download(repo_id=source, filename="model.safetensors")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg, load_file(str(weights_path))


def build_astrom3(module, cfg: Dict, classification: bool, overrides: Optional[Dict] = None):
    kwargs = dict(cfg)
    kwargs.update(overrides or {})
    kwargs["classification"] = classification
    kwargs.setdefault("num_classes", config.NUM_CLASSES)
    keep = ("classification", "num_classes", "hidden_dim", "fusion", "seq_len", "p_enc_in", "p_d_model",
            "p_dropout", "p_factor", "p_output_attention", "p_n_heads", "p_d_ff", "p_activation",
            "p_e_layers", "s_dropout", "s_conv_channels", "s_kernel_size", "s_mp_kernel_size",
            "m_input_dim", "m_hidden_dim", "m_dropout")
    return module.AstroM3(**{k: kwargs[k] for k in keep})


def load_clip_encoder(source: str = config.CLIP_MODEL_NAME, root: Path = ASTROM3_DIR):
    module = import_astrom3(root)
    cfg, weights = load_pretrained_files(source)
    model = build_astrom3(module, cfg, classification=False)
    result = model.load_state_dict(weights, strict=True)
    info = {"n_tensors": len(weights), "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys), "config": cfg}
    return model, info


def build_finetune_model(source: str = config.CLIP_MODEL_NAME, root: Path = ASTROM3_DIR,
                         hp: Dict = FINETUNE_HP, num_classes: int = config.NUM_CLASSES):
    module = import_astrom3(root)
    cfg, weights = load_pretrained_files(source)
    overrides = {k: hp[k] for k in ("fusion", "p_dropout", "s_dropout", "m_dropout")}
    overrides["num_classes"] = num_classes
    model = build_astrom3(module, cfg, classification=True, overrides=overrides)
    result = model.load_state_dict(weights, strict=False)
    missing = list(result.missing_keys)
    if any(not k.startswith("fc.") for k in missing) or result.unexpected_keys:
        raise RuntimeError(f"Pesos incompatibles. Faltan: {missing} | Sobran: {list(result.unexpected_keys)}")
    return model, {"n_tensors": len(weights), "new_layers": missing, "config": cfg}


def crop_or_pad(photometry: torch.Tensor, seq_len: int = SEQ_LEN, random_crop: bool = False,
                generator: Optional[torch.Generator] = None):
    T = photometry.shape[0]
    mask = torch.ones(seq_len, dtype=torch.float32)
    if T > seq_len:
        if random_crop:
            start = int(torch.randint(0, T - seq_len + 1, (1,), generator=generator))
        else:
            start = (T - seq_len) // 2
        photometry = photometry[start:start + seq_len]
    else:
        mask[T:] = 0
        photometry = F.pad(photometry, (0, 0, 0, seq_len - T), value=0)
    return photometry, mask


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = np.stack([np.asarray(v, dtype=np.float32) for v in x])
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


class CachedSplit(Dataset):

    def __init__(self, hf_split):
        self.photometry, self.spectra, self.metadata, self.labels = [], [], [], []
        rows = hf_split.with_format("numpy") if hasattr(hf_split, "with_format") else hf_split
        for row in rows:
            self.photometry.append(_to_tensor(row["photometry"]))
            self.spectra.append(_to_tensor(row["spectra"]))
            self.metadata.append(_to_tensor(row["metadata"]).flatten())
            self.labels.append(int(row["label"]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        return {"photometry": self.photometry[idx], "spectra": self.spectra[idx],
                "metadata": self.metadata[idx], "label": self.labels[idx]}


class ClipCollate:

    def __init__(self, seq_len: int = SEQ_LEN, random_crop: bool = False, seed: int = config.SEED):
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        phot, masks = zip(*(crop_or_pad(b["photometry"], self.seq_len, self.random_crop, self.generator)
                            for b in batch))
        return {
            "photometry": torch.stack(phot),
            "photometry_mask": torch.stack(masks),
            "spectra": torch.stack([b["spectra"] for b in batch]),
            "metadata": torch.stack([b["metadata"] for b in batch]),
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }


def make_clip_loader(split, train: bool, batch_size: int = FINETUNE_HP["batch_size"],
                     seed: int = config.SEED) -> DataLoader:
    dataset = split if isinstance(split, Dataset) else CachedSplit(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                      collate_fn=ClipCollate(random_crop=train, seed=seed),
                      generator=torch.Generator().manual_seed(seed) if train else None)


def forward_fn_clip(model, batch):
    logits = model(batch["photometry"], batch["photometry_mask"], batch["spectra"], batch["metadata"])
    return logits, batch["label"]


@torch.no_grad()
def extract_clip_embeddings(model, loader, device: str) -> Dict[str, np.ndarray]:
    model.eval().to(device)
    out = {"fotometria": [], "espectros": [], "metadatos": [], "labels": []}
    for batch in loader:
        batch = engine._move_batch_to_device(batch, device)
        p, s, m = model.get_embeddings(batch["photometry"], batch["photometry_mask"],
                                       batch["spectra"], batch["metadata"])
        out["fotometria"].append(p.cpu().numpy())
        out["espectros"].append(s.cpu().numpy())
        out["metadatos"].append(m.cpu().numpy())
        out["labels"].append(batch["label"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def count_trainable(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def finetune(model: nn.Module, train_loader, val_loader, device: str, hp: Dict = FINETUNE_HP,
             class_names: Optional[List[str]] = None, verbose: bool = True,
             forward_fn: Callable = forward_fn_clip) -> Dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], betas=(hp["beta1"], hp["beta2"]),
                                 weight_decay=hp["weight_decay"])
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=hp["scheduler_factor"],
                                                         patience=hp["scheduler_patience"])
    warmup = None
    if hp["warmup_epochs"] > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0,
                                                   total_iters=hp["warmup_epochs"])

    history = []
    best = {"f1": -1.0, "epoch": 0, "state": None}
    best_loss, bad_epochs = None, 0
    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_res = engine.train_one_epoch(model, train_loader, optimizer, device, forward_fn)
        val = engine.evaluate(model, val_loader, device, forward_fn, class_names)
        if warmup is not None and epoch <= hp["warmup_epochs"]:
            warmup.step()
        else:
            plateau.step(val["loss"])
        history.append({"epoch": epoch, "lr": lr_now,
                        "train_loss": train_res.loss, "train_acc": train_res.accuracy,
                        "train_macro_f1": train_res.macro_f1,
                        "val_loss": val["loss"], "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        if verbose:
            print(f"[{epoch:02d}/{hp['epochs']}] lr={lr_now:.2e} train_loss={train_res.loss:.4f} "
                  f"train_acc={train_res.accuracy:.3f} | val_loss={val['loss']:.4f} "
                  f"val_acc={val['accuracy']:.3f} val_macroF1={val['macro_f1']:.3f}")
        if val["macro_f1"] > best["f1"]:
            best = {"f1": val["macro_f1"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if best_loss is None or val["loss"] < best_loss:
            best_loss, bad_epochs = val["loss"], 0
        else:
            bad_epochs += 1
        if bad_epochs >= hp["early_stopping_patience"]:
            if verbose:
                print(f"Parada temprana en la época {epoch}")
            break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "best_val_macro_f1": best["f1"],
            "best_state": best["state"], "last_state": last_state, "stopped_epoch": len(history),
            "train_seconds": time.time() - t0, "n_parameters": count_trainable(model)}