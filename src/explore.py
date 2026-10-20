# -*- coding: utf-8 -*-
"""Exploración de datos pedida por el tutor (deuda del Entregable 1).

Todas las funciones leen splits reales de Hugging Face (los que devuelve
datasets.load_subset) y devuelven tablas (pandas) con lo MEDIDO; no asumen
nada sobre el significado de los canales: sirven justamente para averiguarlo.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

PHOT, SPEC, META, LABEL = "photometry", "spectra", "metadata", "label"


def _sample_indices(n_total: int, n: Optional[int], seed: int = config.SEED) -> np.ndarray:
    if n is None or n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.RandomState(seed).choice(n_total, size=n, replace=False))


def class_distribution(splits: Dict[str, object], class_names: List[str] = config.CLASS_NAMES) -> pd.DataFrame:
    """splits: {'train': hf_split, 'validation': ..., 'test': ...}. Filas = clases."""
    cols = {}
    for name, ds in splits.items():
        labels = np.asarray(ds[LABEL])
        cols[name] = np.bincount(labels, minlength=len(class_names))
    df = pd.DataFrame(cols, index=class_names)
    df["total"] = df.sum(axis=1)
    for name in splits:
        df[f"{name}_%"] = (100 * df[name] / df[name].sum()).round(1)
    return df


def photometry_lengths(hf_split, n: Optional[int] = None) -> pd.Series:
    """Número de puntos de cada curva de luz."""
    idx = _sample_indices(len(hf_split), n)
    return pd.Series([len(hf_split[int(i)][PHOT]) for i in idx], name="longitud")


def photometry_channel_stats(hf_split, n: Optional[int] = 2000) -> pd.DataFrame:
    """Por canal de la fotometría: media/desv./mín/máx sobre todos los puntos, y qué fracción de
    los objetos lo tiene CONSTANTE a lo largo de la curva (esperable en canales auxiliares
    derivados de un valor por objeto, p. ej. período o amplitud)."""
    idx = _sample_indices(len(hf_split), n)
    pts, const_flags = [], []
    for i in idx:
        arr = np.asarray(hf_split[int(i)][PHOT], dtype=np.float64)       # (T, C)
        pts.append(arr)
        const_flags.append(arr.std(axis=0) < 1e-8)
    allp = np.concatenate(pts, axis=0)
    const = np.mean(np.stack(const_flags), axis=0)
    return pd.DataFrame({
        "media": allp.mean(0), "desv": allp.std(0), "min": allp.min(0), "max": allp.max(0),
        "frac_objetos_constante_en_la_curva": const,
        "n_nan": np.isnan(allp).sum(0),
    }).rename_axis("canal")


def spectra_channel_stats(hf_split, n: Optional[int] = 1000) -> pd.DataFrame:
    """Por canal del espectro (3 canales): escala, ceros y NaN. Sirve para diagnosticar la CNN
    (¿escalas muy distintas entre canales? ¿espectros sin normalizar?)."""
    idx = _sample_indices(len(hf_split), n)
    arrs = np.stack([np.asarray(hf_split[int(i)][SPEC], dtype=np.float64) for i in idx])   # (n,3,L)
    return pd.DataFrame({
        "media": arrs.mean(axis=(0, 2)), "desv": arrs.std(axis=(0, 2)),
        "min": arrs.min(axis=(0, 2)), "max": arrs.max(axis=(0, 2)),
        "media_de_la_desv_por_objeto": arrs.std(axis=2).mean(axis=0),
        "frac_ceros": (arrs == 0).mean(axis=(0, 2)),
        "n_nan": np.isnan(arrs).sum(axis=(0, 2)),
    }).rename_axis("canal")


def metadata_stats(hf_split, n: Optional[int] = None) -> pd.DataFrame:
    """Por columna de metadatos (34): escala, NaN y si es constante."""
    idx = _sample_indices(len(hf_split), n)
    m = np.stack([np.asarray(hf_split[int(i)][META], dtype=np.float64).ravel() for i in idx])
    return pd.DataFrame({
        "media": m.mean(0), "desv": m.std(0), "min": m.min(0), "max": m.max(0),
        "n_nan": np.isnan(m).sum(0), "constante": m.std(0) < 1e-12,
    }).rename_axis("columna")


def first_example_per_class(hf_split, n_classes: int = config.NUM_CLASSES) -> Dict[int, int]:
    """Índice del primer objeto de cada clase (para graficar ejemplos)."""
    labels = np.asarray(hf_split[LABEL])
    return {c: int(np.where(labels == c)[0][0]) for c in range(n_classes) if (labels == c).any()}
