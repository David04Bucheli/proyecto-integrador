# -*- coding: utf-8 -*-
"""Prueba de humo del flujo del Entregable 2 (runs, explore, reporting, fusion encadenados)
con datos SINTÉTICOS. No usa el dataset real ni produce resultados: solo valida que
el código corre de punta a punta y que los archivos se guardan y recargan bien.

Ejecutar desde la raíz del repositorio:
    python tests/test_entregable2.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from src import config, datasets, engine, embeddings, fusion, models, reporting, runs, explore

torch.manual_seed(config.SEED)
rng = np.random.RandomState(config.SEED)
OK = True


def check(name, cond):
    global OK
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    OK &= bool(cond)


class FakeHF:
    """Imita lo mínimo de un split de Hugging Face: len, split[i] -> dict y split['columna'] -> lista."""
    def __init__(self, n):
        self.rows = []
        for i in range(n):
            T = int(rng.randint(10, 40))
            ph = rng.randn(T, 9)
            ph[:, 3:] = rng.randn(6)            # canales 3..8 constantes por objeto (como se sospecha del dataset real)
            self.rows.append({"photometry": ph.tolist(), "spectra": rng.randn(3, 2575).tolist(),
                              "metadata": rng.randn(34).tolist(), "label": int(i % config.NUM_CLASSES)})

    def __len__(self): return len(self.rows)
    def __getitem__(self, k): return self.rows[k] if isinstance(k, int) else [r[k] for r in self.rows]


train, val, test = FakeHF(60), FakeHF(30), FakeHF(30)
FWD = {"fotometria": engine.forward_fn_photometry, "espectros": engine.forward_fn_spectra,
       "metadatos": engine.forward_fn_metadata}


def build(name):
    if name == "fotometria": return models.UnimodalClassifier(models.PhotometryGRUEncoder(input_dim=9))
    if name == "espectros": return models.UnimodalClassifier(models.SpectraCNNEncoder(in_channels=3))
    return models.UnimodalClassifier(models.MetadataMLPEncoder(input_dim=34))


tmp = Path(tempfile.mkdtemp())
tr_loader = datasets.make_dataloader(train, batch_size=16, shuffle=True)
loaders = {s: datasets.make_dataloader(d, batch_size=16, shuffle=False) for s, d in
           (("train", train), ("validation", val), ("test", test))}

print("=== 1) Exploración sobre un split simulado ===")
dist = explore.class_distribution({"train": train, "validation": val, "test": test})
check("distribución: 10 filas y total correcto", len(dist) == 10 and int(dist["train"].sum()) == 60)
lens = explore.photometry_lengths(train)
check("longitudes: una por objeto", len(lens) == 60 and lens.min() >= 10)
ps = explore.photometry_channel_stats(train, n=None)
check("canales 3..8 detectados como constantes por objeto", (ps["frac_objetos_constante_en_la_curva"].values[3:] > 0.99).all())
check("canales 0..2 NO constantes", (ps["frac_objetos_constante_en_la_curva"].values[:3] < 0.01).all())
check("stats de espectro: 3 canales", len(explore.spectra_channel_stats(train, n=20)) == 3)
check("stats de metadatos: 34 columnas", len(explore.metadata_stats(train)) == 34)
check("un ejemplo por clase", len(explore.first_example_per_class(train)) == 10)

print("=== 2) Entrenamiento con checkpoints, guardado y recarga ===")
MODELS, PRED, VAL_F1 = {}, {}, {}
for name in FWD:
    m = build(name)
    fr = runs.fit_with_checkpoints(m, tr_loader, loaders["validation"], FWD[name], num_epochs=2,
                                   class_names=config.CLASS_NAMES, verbose=False)
    check(f"{name}: historial de 2 épocas, mejor época válida", len(fr["history"]) == 2 and fr["best_epoch"] in (1, 2))
    check(f"{name}: parámetros contados", fr["n_parameters"] > 0)
    path = tmp / f"{name}.pt"
    runs.save_checkpoint(path, fr, extra={"seed": config.SEED})
    fresh = build(name)
    runs.load_checkpoint_into(fresh, path, which="last")
    a = runs.predict_proba(m, loaders["test"], "cpu", FWD[name])
    b = runs.predict_proba(fresh, loaders["test"], "cpu", FWD[name])
    check(f"{name}: modelo recargado predice igual que el original", np.allclose(a["probs"], b["probs"], atol=1e-6))
    check(f"{name}: probabilidades suman 1", np.allclose(a["probs"].sum(1), 1, atol=1e-5))
    runs.load_checkpoint_into(fresh, path, which="best")
    MODELS[name] = fresh
    PRED[name] = b
    VAL_F1[name] = fusion.summarize_predictions(
        *(lambda o: (o["y_true"], o["y_pred"]))(runs.predict_proba(fresh, loaders["validation"], "cpu", FWD[name])))["macro_f1"]
    npz = tmp / f"{name}_pred.npz"
    runs.save_predictions(npz, b)
    check(f"{name}: predicciones guardadas y recargadas", np.array_equal(runs.load_predictions(npz)["y_pred"], b["y_pred"]))

print("=== 3) Votación sobre predicciones alineadas ===")
fusion.check_alignment([PRED[n]["y_true"] for n in FWD])
check("etiquetas alineadas (shuffle=False)", True)
prio = fusion.rank_by_validation(VAL_F1)
for rule in fusion.TIE_BREAKS:
    v = fusion.hard_vote({n: PRED[n]["y_pred"] for n in FWD}, tie_break=rule, priority=prio,
                         probs={n: PRED[n]["probs"] for n in FWD})
    m = fusion.summarize_predictions(PRED["fotometria"]["y_true"], v["y_pred"], config.CLASS_NAMES)
    check(f"votación con desempate '{rule}' produce métricas", 0 <= m["accuracy"] <= 1 and m["confusion_matrix"].shape == (10, 10))
check("voto suave produce predicciones", fusion.soft_vote({n: PRED[n]["probs"] for n in FWD}).shape == (30,))

print("=== 4) Gráficos ===")
fig = reporting.plot_confusion_matrix(m["confusion_matrix"], title="prueba", path=str(tmp / "cm.png"))
check("matriz de confusión guardada como imagen", (tmp / "cm.png").exists())
hist = runs.load_checkpoint_into(build("metadatos"), tmp / "metadatos.pt")["history"]
reporting.plot_history({"metadatos": hist}, path=str(tmp / "hist.png"))
check("curvas de entrenamiento guardadas", (tmp / "hist.png").exists())
check("tabla resumen", reporting.summary_table({"x": m}).shape == (1, 2))
check("tabla por clase", reporting.per_class_table({"x": m}).shape == (10, 1))

print("=== 5) Embeddings alineados con las mismas funciones del Entregable 1 ===")
E = {n: embeddings.extract_embeddings(MODELS[n], loaders["test"], "cpu", FWD[n]) for n in FWD}
fusion.check_alignment([E[n]["labels"] for n in FWD])
fused = embeddings.concatenate_modalities(E["fotometria"]["embeddings"], E["espectros"]["embeddings"],
                                          E["metadatos"]["embeddings"], E["fotometria"]["labels"])
check("embedding fusionado (N, 192)", fused["embeddings"].shape == (30, 3 * config.EMBEDDING_DIM))

print("\nRESULTADO:", "TODO OK (datos sintéticos)" if OK else "HAY FALLAS")
sys.exit(0 if OK else 1)
