# -*- coding: utf-8 -*-
"""Fusión por votación (Entregable 2).

Trabaja sobre las predicciones de los tres modelos unimodales ya entrenados,
evaluadas sobre el MISMO split y en el MISMO orden (loaders con shuffle=False).
No entrena nada: es fusión tardía (a nivel de decisión).

Regla principal: mayoría 2-de-3 (voto duro). Con 10 clases puede pasar que los
tres modelos voten clases distintas; ahí hay que decidir con una regla de
desempate, que se elige explícitamente (ver TIE_BREAKS) y se reporta.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from . import config

TIE_BREAKS = ("priority", "confidence", "error")
NO_DECISION = -1   # etiqueta usada por tie_break="error": cuenta como predicción incorrecta


def check_alignment(y_true_list: Sequence[np.ndarray]) -> None:
    """Las tres modalidades deben haberse evaluado sobre los mismos objetos y en el mismo orden."""
    ref = np.asarray(y_true_list[0])
    for i, y in enumerate(y_true_list[1:], start=1):
        y = np.asarray(y)
        if y.shape != ref.shape or not np.array_equal(y, ref):
            raise ValueError(
                f"Las etiquetas de la modalidad {i} no coinciden con las de la modalidad 0: "
                "los loaders deben usar shuffle=False y el mismo split.")


def rank_by_validation(val_macro_f1: Dict[str, float]) -> List[str]:
    """Ordena modalidades de mejor a peor macro-F1 de VALIDACIÓN (nunca de test)."""
    return sorted(val_macro_f1, key=lambda k: val_macro_f1[k], reverse=True)


def hard_vote(preds: Dict[str, np.ndarray], tie_break: str = "priority",
              priority: Optional[List[str]] = None,
              probs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Mayoría 2-de-3.

    preds: {modalidad: array (N,) de clases predichas}; exactamente 3 modalidades.
    tie_break (solo si las 3 votan clases distintas):
      - "priority":   gana el voto de la primera modalidad de `priority` (p. ej. la mejor en validación).
      - "confidence": gana el voto del modelo con mayor probabilidad softmax máxima (requiere `probs`).
      - "error":      no se decide (NO_DECISION = -1) y cuenta como error.
    Devuelve y_pred y `case` por objeto: 3 = unánime, 2 = mayoría 2-de-3, 1 = los tres difieren.
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break debe ser uno de {TIE_BREAKS}")
    names = list(preds)
    if len(names) != 3:
        raise ValueError("La votación 2-de-3 necesita exactamente 3 modalidades.")
    P = np.stack([np.asarray(preds[n]) for n in names], axis=0)      # (3, N)
    N = P.shape[1]
    if tie_break == "priority":
        if not priority or set(priority) != set(names):
            raise ValueError("tie_break='priority' requiere `priority` con las 3 modalidades.")
    if tie_break == "confidence":
        if probs is None:
            raise ValueError("tie_break='confidence' requiere `probs`.")
        conf = np.stack([np.asarray(probs[n]).max(axis=1) for n in names], axis=0)   # (3, N)

    final = np.full(N, NO_DECISION, dtype=np.int64)
    case = np.ones(N, dtype=np.int64)
    a, b, c = P[0], P[1], P[2]

    unanimous = (a == b) & (b == c)
    ab, ac, bc = (a == b), (a == c), (b == c)
    majority = (ab | ac | bc) & ~unanimous
    final[unanimous] = a[unanimous]
    case[unanimous] = 3
    case[majority] = 2
    # en un 2-de-3 el voto mayoritario es el de cualquier par que coincide
    m_ab = majority & ab
    m_ac = majority & ac & ~ab
    m_bc = majority & bc & ~ab & ~ac
    final[m_ab] = a[m_ab]
    final[m_ac] = a[m_ac]
    final[m_bc] = b[m_bc]

    three_way = ~(unanimous | majority)
    if three_way.any():
        if tie_break == "priority":
            first = names.index(priority[0])
            final[three_way] = P[first][three_way]
        elif tie_break == "confidence":
            winner = conf.argmax(axis=0)                              # (N,) índice de modalidad
            final[three_way] = P[winner, np.arange(N)][three_way]
        # "error": se queda en NO_DECISION
    return {"y_pred": final, "case": case, "modalities": np.array(names)}


def soft_vote(probs: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Referencia complementaria (no es la regla 2-de-3 prometida): promedio, opcionalmente
    ponderado, de las probabilidades softmax de las modalidades. Devuelve y_pred (N,)."""
    names = list(probs)
    w = np.array([1.0 if weights is None else weights[n] for n in names], dtype=float)
    w = w / w.sum()
    avg = sum(wi * np.asarray(probs[n]) for wi, n in zip(w, names))
    return avg.argmax(axis=1)


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None) -> Dict:
    """Mismas métricas que engine.evaluate, calculadas desde y_true/y_pred. Predicciones = -1
    (sin decisión) cuentan como errores y no aparecen en la matriz de confusión."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(config.NUM_CLASSES))
    names = class_names or [str(i) for i in labels]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                 labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {names[i]: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "report_text": classification_report(y_true, y_pred, labels=labels, target_names=names,
                                             zero_division=0),
    }


def agreement_analysis(y_true: np.ndarray, preds: Dict[str, np.ndarray], vote: Dict) -> Dict:
    """Cuántas veces coinciden los modelos y cuánto acierta la votación en cada caso;
    más el 'oráculo' (cota superior: algún modelo acierta) y la exactitud de cada modalidad."""
    y_true = np.asarray(y_true)
    case, final = vote["case"], vote["y_pred"]
    correct = {n: (np.asarray(p) == y_true) for n, p in preds.items()}
    any_correct = np.any(np.stack(list(correct.values())), axis=0)
    out = {"n": int(len(y_true)),
           "oracle_any_model_correct": float(any_correct.mean()),
           "per_modality_accuracy": {n: float(c.mean()) for n, c in correct.items()},
           "cases": {}}
    for code, label in ((3, "unánime"), (2, "mayoría 2-de-3"), (1, "los 3 difieren")):
        m = case == code
        out["cases"][label] = {
            "n": int(m.sum()), "fraccion": float(m.mean()),
            "accuracy_votacion": float((final[m] == y_true[m]).mean()) if m.any() else None,
        }
    return out
