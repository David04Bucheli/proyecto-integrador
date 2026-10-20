# -*- coding: utf-8 -*-
"""Pruebas de la votación 2-de-3 (src/fusion.py) con casos construidos a mano.
No usa el dataset real ni resultados reales: verifica la lógica.

Ejecutar desde la raíz del repositorio:
    python tests/test_fusion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import fusion

OK = True


def check(name, cond):
    global OK
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    OK &= bool(cond)


# 6 objetos con un caso cada uno: foto, espectro, metadatos
#   0: unánime (1,1,1)                 -> 1
#   1: mayoría foto=meta (2,5,2)       -> 2
#   2: mayoría foto=espectro (3,3,7)   -> 3
#   3: mayoría espectro=meta (0,4,4)   -> 4
#   4: los 3 difieren (1,2,3)          -> depende del desempate
#   5: los 3 difieren (9,8,7)          -> depende del desempate
preds = {
    "foto": np.array([1, 2, 3, 0, 1, 9]),
    "espectro": np.array([1, 5, 3, 4, 2, 8]),
    "meta": np.array([1, 2, 7, 4, 3, 7]),
}
probs = {  # solo importa la confianza máxima de cada modelo en los objetos 4 y 5
    "foto": np.eye(10)[[1, 2, 3, 0, 1, 9]] * 0.5 + 0.05,       # conf 0.55
    "espectro": np.eye(10)[[1, 5, 3, 4, 2, 8]] * 0.9 + 0.01,   # conf 0.91
    "meta": np.eye(10)[[1, 2, 7, 4, 3, 7]] * 0.3 + 0.07,       # conf 0.37
}

print("=== 1) Mayoría y casos ===")
v = fusion.hard_vote(preds, tie_break="priority", priority=["foto", "meta", "espectro"])
check("unánime y mayorías correctas", list(v["y_pred"][:4]) == [1, 2, 3, 4])
check("códigos de caso", list(v["case"]) == [3, 2, 2, 2, 1, 1])

print("=== 2) Desempates ===")
check("priority=foto -> voto de foto en 3 distintos", list(v["y_pred"][4:]) == [1, 9])
v2 = fusion.hard_vote(preds, tie_break="priority", priority=["espectro", "foto", "meta"])
check("priority=espectro -> voto de espectro", list(v2["y_pred"][4:]) == [2, 8])
v3 = fusion.hard_vote(preds, tie_break="confidence", probs=probs)
check("confidence -> modelo más seguro (espectro)", list(v3["y_pred"][4:]) == [2, 8])
v4 = fusion.hard_vote(preds, tie_break="error")
check("error -> sin decisión (-1)", list(v4["y_pred"][4:]) == [fusion.NO_DECISION] * 2)
check("el desempate no altera objetos con mayoría", list(v4["y_pred"][:4]) == [1, 2, 3, 4])

print("=== 3) Validaciones ===")
for bad in (lambda: fusion.hard_vote(preds, tie_break="priority"),
            lambda: fusion.hard_vote(preds, tie_break="confidence"),
            lambda: fusion.hard_vote({"a": preds["foto"], "b": preds["meta"]}, tie_break="error"),
            lambda: fusion.hard_vote(preds, tie_break="otro")):
    try:
        bad(); raised = False
    except ValueError:
        raised = True
    check("lanza ValueError ante uso inválido", raised)

y = np.array([0, 1, 2])
try:
    fusion.check_alignment([y, y.copy(), np.array([0, 2, 1])]); raised = False
except ValueError:
    raised = True
check("check_alignment detecta orden distinto", raised)
fusion.check_alignment([y, y.copy(), y.copy()])
check("check_alignment acepta etiquetas iguales", True)

print("=== 4) Métricas y análisis ===")
y_true = np.array([1, 2, 3, 4, 1, 9])
m = fusion.summarize_predictions(y_true, v["y_pred"], class_names=list("ABCDEFGHIJ"))
check("votación (priority=foto) acierta los 6", abs(m["accuracy"] - 1.0) < 1e-9)
m4 = fusion.summarize_predictions(y_true, v4["y_pred"])
check("'sin decisión' cuenta como error (acc=4/6)", abs(m4["accuracy"] - 4 / 6) < 1e-9)
check("matriz de confusión 10x10", m["confusion_matrix"].shape == (10, 10))
ag = fusion.agreement_analysis(y_true, preds, v)
check("casos suman N", sum(c["n"] for c in ag["cases"].values()) == 6)
check("oráculo >= votación", ag["oracle_any_model_correct"] >= m["accuracy"] - 1e-9)

print("=== 5) Voto suave (referencia) ===")
sv = fusion.soft_vote(probs)
check("soft_vote devuelve (N,)", sv.shape == (6,))

print("\nRESULTADO:", "TODO OK" if OK else "HAY FALLAS")
sys.exit(0 if OK else 1)
