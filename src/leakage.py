import hashlib
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

COLUMNS = {"metadatos": "metadata", "fotometria": "photometry", "espectros": "spectra"}
KINDS = ("metadatos", "fotometria", "espectros")
DECIMALS = (3, 4, 6)
MIN_RECALL = 0.99


def _as_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = np.stack([np.asarray(v, dtype=np.float64) for v in value])
    return np.asarray(value, dtype=np.float64)


def _rows(split, column: str) -> Iterable:
    if hasattr(split, "select_columns"):
        split = split.select_columns([column]).with_format("numpy")
    for row in split:
        yield row[column]


def _core(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "fotometria":
        return arr[:, :3]
    if kind == "espectros":
        return arr[:2]
    return arr.reshape(-1)


def _hash(arr: np.ndarray, decimals: int) -> str:
    r = np.round(arr, decimals) + 0.0
    return hashlib.sha1(str(r.shape).encode() + np.ascontiguousarray(r).tobytes()).hexdigest()


def split_fingerprints(split, kind: str, decimals: Sequence[int] = DECIMALS) -> Dict[int, List[str]]:
    out = {d: [] for d in decimals}
    for value in _rows(split, COLUMNS[kind]):
        core = _core(_as_array(value), kind)
        for d in decimals:
            out[d].append(_hash(core, d))
    return out


def audit_table(reference_train, targets: Dict[str, object], kinds: Sequence[str] = KINDS,
                decimals: Sequence[int] = DECIMALS, positive_control: str = "train",
                min_recall: float = MIN_RECALL) -> pd.DataFrame:
    rows = []
    for kind in kinds:
        ref = split_fingerprints(reference_train, kind, decimals)
        tgt = {name: split_fingerprints(split, kind, decimals) for name, split in targets.items()}
        for d in decimals:
            ref_set = set(ref[d])
            row = {"huella": kind, "decimales": d, "filas referencia": len(ref[d]),
                   "huellas únicas referencia": len(ref_set)}
            for name in targets:
                tset = set(tgt[name][d])
                row[f"únicas {name}"] = len(tset)
                row[f"coincidencias {name}"] = len(ref_set & tset)
            pc = set(tgt[positive_control][d])
            row["control positivo (recuperación)"] = len(ref_set & pc) / max(len(pc), 1)
            row["huella válida"] = row["control positivo (recuperación)"] >= min_recall
            rows.append(row)
    return pd.DataFrame(rows).set_index(["huella", "decimales"])


def decide(table: pd.DataFrame, checked: Sequence[str] = ("validation", "test")) -> Dict:
    valid = table[table["huella válida"]]
    if valid.empty:
        return {"ok": None, "valid": [],
                "message": "Ninguna huella recupera el control positivo: la auditoría no puede verificar la separación."}
    leaks = {f"{k}@{d}": {s: int(valid.loc[(k, d), f"coincidencias {s}"]) for s in checked}
             for k, d in valid.index if any(valid.loc[(k, d), f"coincidencias {s}"] > 0 for s in checked)}
    names = [f"{k}@{d}" for k, d in valid.index]
    if leaks:
        return {"ok": False, "valid": names, "leaks": leaks,
                "message": "Hay objetos de validación o prueba en el conjunto de preentrenamiento."}
    return {"ok": True, "valid": names,
            "message": "Las huellas válidas no encuentran objetos de validación ni de prueba en el preentrenamiento."}