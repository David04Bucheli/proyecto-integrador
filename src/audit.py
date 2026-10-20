# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import config


# n° de decimales al redondear los metadatos
FINGERPRINT_DECIMALS = 6


@dataclass
class LeakageAuditReport:
    n_train_full: int
    n_test_sub50: int
    n_overlap: int
    overlapping_ids: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.n_overlap == 0

    def __str__(self):
        status = "SIN FUGA DETECTADA" if self.is_clean else "FUGA DETECTADA"
        lines = [
            "---- Auditoría de fuga de datos (AstroM3-CLIP) ----",
            f"Objetos en full/train:      {self.n_train_full}",
            f"Objetos en sub50/test:      {self.n_test_sub50}",
            f"Objetos en la intersección: {self.n_overlap}",
            f"Resultado: {status}",
        ]
        if not self.is_clean:
            preview = self.overlapping_ids[:5]
            lines.append(f"Ejemplos de objetos en fuga (máx 5): {preview}")
        return "\n".join(lines)


#hace al vector de metadata una tupla hasheable (el identificador del objeto)
def _metadata_fingerprint(metadata_row: Sequence[float],
                           decimals: int = FINGERPRINT_DECIMALS):
    return tuple(round(float(v), decimals) for v in metadata_row)


#calcula la huella de cada fila de un split
def _fingerprint_split(hf_split, metadata_key: str = "metadata",
                        decimals: int = FINGERPRINT_DECIMALS) -> List[tuple]:
    return [
        _metadata_fingerprint(row, decimals=decimals)
        for row in hf_split[metadata_key]
    ]

#ejecuta el audit
def run_leakage_audit(
    dataset_name: str = config.DATASET_NAME,
    seed: int = config.SEED
) -> LeakageAuditReport:

    from datasets import load_dataset

    full = load_dataset(dataset_name, name=f"full_{seed}")
    sub50 = load_dataset(dataset_name, name=f"sub50_{seed}")

    train_fps = _fingerprint_split(full["train"])
    test_fps = _fingerprint_split(sub50["test"])
    train_set, test_set = set(train_fps), set(test_fps)
    overlap = train_set & test_set

    return LeakageAuditReport(
        n_train_full=len(train_set),
        n_test_sub50=len(test_set),
        n_overlap=len(overlap),
        overlapping_ids=sorted(overlap)[:50],
    )


#chequeo de prueba entre splits train y test de sub10
def run_self_check(subset_name: str = "sub10", seed: int = config.SEED,
                    dataset_name: str = config.DATASET_NAME) -> LeakageAuditReport:

    from datasets import load_dataset

    small = load_dataset(dataset_name, name=f"{subset_name}_{seed}")

    train_fps = _fingerprint_split(small["train"])
    test_fps = _fingerprint_split(small["test"])
    train_set, test_set = set(train_fps), set(test_fps)
    overlap = train_set & test_set

    return LeakageAuditReport(
        n_train_full=len(train_set),
        n_test_sub50=len(test_set),
        n_overlap=len(overlap),
        overlapping_ids=sorted(overlap)[:50],
    )


#para ver las columnas disponibles y una fila de ejemplo
def inspect_available_columns(dataset_name: str = config.DATASET_NAME,
                               seed: int = config.SEED):

    from datasets import load_dataset

    full = load_dataset(dataset_name, name=f"full_{seed}")
    print("Columnas disponibles:", list(full["train"].features.keys()))
    print("\nPrimer ejemplo:")
    example = full["train"][0]
    for k, v in example.items():
        v_repr = type(v).__name__
        try:
            v_repr += f" (len={len(v)})"
        except TypeError:
            v_repr += f" = {v}"
        print(f"  {k}: {v_repr}")
