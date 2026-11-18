import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src import compare, concat, config, leakage, runs, transfer

torch.manual_seed(config.SEED)
rng = np.random.RandomState(config.SEED)
OK = True


def check(name, cond):
    global OK
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    OK &= bool(cond)


class FakeHF:
    def __init__(self, n, min_len=50, max_len=400):
        self.rows = []
        for i in range(n):
            T = int(rng.randint(min_len, max_len))
            self.rows.append({"photometry": rng.randn(T, 9).tolist(), "spectra": rng.randn(3, 2575).tolist(),
                              "metadata": rng.randn(34).tolist(), "label": int(i % config.NUM_CLASSES)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        return self.rows[k] if isinstance(k, int) else [r[k] for r in self.rows]

    def __iter__(self):
        return iter(self.rows)


print("=== 1) Concatenación: cabeza MLP y búsqueda en validación ===")
centers = rng.randn(config.NUM_CLASSES, 192) * 3


def make_emb(n):
    y = np.arange(n) % config.NUM_CLASSES
    return (centers[y] + rng.randn(n, 192)).astype(np.float32), y


Xtr, ytr = make_emb(400)
Xva, yva = make_emb(100)
Xte, yte = make_emb(100)
gs = concat.grid_search(Xtr, ytr, Xva, yva, grid={"hidden_dim": [0, 32], "dropout": [0.1], "weight_decay": [0.0]},
                        max_epochs=15, patience=5, verbose=False)
check("búsqueda devuelve una fila por configuración", len(gs["table"]) == 2)
check("la mejor configuración es la de mayor macro-F1 en validación",
      gs["best"]["best_val_macro_f1"] == max(r["val_macro_f1"] for r in gs["table"]))
pred = concat.predict_proba_head(gs["best"], Xte, yte)
check("datos separables: macro-F1 de prueba > 0.9", compare.macro_f1(yte, pred["y_pred"]) > 0.9)
check("probabilidades suman 1", np.allclose(pred["probs"].sum(1), 1, atol=1e-5))
check("el escalado usa solo la media de entrenamiento", np.allclose(gs["best"]["scaler"].mean, Xtr.mean(0, keepdims=True)))
sub = concat.take_modalities(Xtr, ["fotometria", "metadatos"], 64)
check("selección de bloques por modalidad", sub.shape == (400, 128) and np.array_equal(sub[:, 64:], Xtr[:, 128:]))

print("=== 2) Auditoría con control positivo (metadatos reestandarizados por configuración) ===")
reference = FakeHF(30)


def restandardize(row):
    r = copy.deepcopy(row)
    r["metadata"] = (np.asarray(r["metadata"]) * 1.3 + 0.2).tolist()
    return r


class Rows(FakeHF):
    def __init__(self, rows):
        self.rows = rows


target_train = Rows([restandardize(r) for r in reference.rows[:20]])
target_val = FakeHF(8)
clean_test = FakeHF(8)
leaky_test = Rows(FakeHF(7).rows + [restandardize(reference.rows[25])])
tbl = leakage.audit_table(reference, {"train": target_train, "validation": target_val, "test": clean_test}, decimals=(4,))
check("la huella de metadatos falla el control positivo", not tbl.loc[("metadatos", 4), "huella válida"])
check("las huellas de fotometría y espectro lo superan",
      tbl.loc[("fotometria", 4), "huella válida"] and tbl.loc[("espectros", 4), "huella válida"])
check("sin fuga: decisión positiva", leakage.decide(tbl)["ok"] is True)
tbl = leakage.audit_table(reference, {"train": target_train, "validation": target_val, "test": leaky_test}, decimals=(4,))
verdict = leakage.decide(tbl)
check("fuga inyectada en prueba: detectada por las huellas válidas", verdict["ok"] is False
      and tbl.loc[("fotometria", 4), "coincidencias test"] == 1)
check("los metadatos no la detectan", tbl.loc[("metadatos", 4), "coincidencias test"] == 0)
only_meta = leakage.audit_table(reference, {"train": target_train, "validation": target_val, "test": leaky_test},
                                kinds=("metadatos",), decimals=(4,))
check("sin huellas válidas: la auditoría no decide", leakage.decide(only_meta)["ok"] is None)

print("=== 3) Comparación por bootstrap pareado ===")
same = compare.paired_bootstrap(yte, pred["y_pred"], pred["y_pred"], n_boot=200)
check("misma predicción: diferencia 0 e intervalo [0, 0]", same["delta_macro_f1"] == 0 and same["ci95_macro_f1"] == (0.0, 0.0))
worse = np.where(np.arange(len(yte)) % 3 == 0, (yte + 1) % config.NUM_CLASSES, yte)
r = compare.paired_bootstrap(yte, yte, worse, n_boot=200)
check("predicción perfecta frente a una peor: intervalo por encima de 0", r["ci95_macro_f1"][0] > 0)

print("=== 4) Recorte y relleno de la fotometría a 200 pasos ===")
long_seq = torch.arange(500 * 9, dtype=torch.float32).reshape(500, 9)
x, m = transfer.crop_or_pad(long_seq)
check("recorte central de 500 a 200", x.shape == (200, 9) and torch.equal(x[0], long_seq[150]) and m.sum() == 200)
x, m = transfer.crop_or_pad(long_seq[:80])
check("relleno de 80 a 200 con máscara", x.shape == (200, 9) and m.sum() == 80 and x[80:].abs().sum() == 0)
g = torch.Generator().manual_seed(0)
x, _ = transfer.crop_or_pad(long_seq, random_crop=True, generator=g)
check("recorte aleatorio contiguo", x.shape == (200, 9) and torch.equal(x[1] - x[0], long_seq[1] - long_seq[0]))

print("=== 5) AstroM3: código oficial, pesos y ajuste fino (modelo pequeño con pesos aleatorios) ===")
root = transfer.fetch_astrom3_code()
module = transfer.import_astrom3(root)
tiny_cfg = {"classification": False, "fusion": "avg", "hidden_dim": 32, "num_classes": 10, "seq_len": 200,
            "p_enc_in": 9, "p_d_model": 16, "p_dropout": 0.1, "p_factor": 1, "p_output_attention": False,
            "p_n_heads": 2, "p_d_ff": 32, "p_activation": "gelu", "p_e_layers": 1,
            "s_dropout": 0.1, "s_conv_channels": [3, 8, 8, 8, 32], "s_kernel_size": 3, "s_mp_kernel_size": 4,
            "m_input_dim": 34, "m_hidden_dim": 32, "m_dropout": 0.1}
tmp = Path(tempfile.mkdtemp())
pre = transfer.build_astrom3(module, tiny_cfg, classification=False)
from safetensors.torch import save_file
save_file({k: v.contiguous() for k, v in pre.state_dict().items()}, str(tmp / "model.safetensors"))
(tmp / "config.json").write_text(json.dumps(tiny_cfg))

enc, info = transfer.load_clip_encoder(str(tmp), root)
check("carga estricta de pesos sin faltantes ni sobrantes", not info["missing"] and not info["unexpected"])
train, val = FakeHF(40), FakeHF(20)
cached_val = transfer.CachedSplit(val)
loader = transfer.make_clip_loader(cached_val, train=False, batch_size=8)
emb = transfer.extract_clip_embeddings(enc, loader, "cpu")
check("embeddings CLIP por modalidad con norma 1", emb["fotometria"].shape == (20, 32)
      and np.allclose(np.linalg.norm(emb["espectros"], axis=1), 1, atol=1e-4))
check("etiquetas en el orden del split", np.array_equal(emb["labels"], np.array(val["label"])))

ft, ft_info = transfer.build_finetune_model(str(tmp), root)
check("solo la capa de clasificación es nueva", sorted(ft_info["new_layers"]) == ["fc.bias", "fc.weight"])
check("los pesos preentrenados se copian", torch.equal(ft.metadata_proj.weight, pre.metadata_proj.weight))
hp = dict(transfer.FINETUNE_HP, epochs=3, warmup_epochs=1, batch_size=8, early_stopping_patience=10)
res = transfer.finetune(ft, transfer.make_clip_loader(train, train=True, batch_size=8), loader, "cpu", hp=hp,
                        class_names=config.CLASS_NAMES, verbose=False)
check("ajuste fino corre 3 épocas y guarda la mejor", len(res["history"]) == 3 and res["best_epoch"] in (1, 2, 3))
check("el ajuste fino modifica los pesos", not torch.equal(ft.metadata_proj.weight.cpu(), pre.metadata_proj.weight))
out = runs.predict_proba(ft, loader, "cpu", transfer.forward_fn_clip)
check("predicción con probabilidades", out["probs"].shape == (20, 10) and np.allclose(out["probs"].sum(1), 1, atol=1e-5))

print("\nRESULTADO:", "TODO OK (datos sintéticos)" if OK else "HAY FALLAS")
sys.exit(0 if OK else 1)