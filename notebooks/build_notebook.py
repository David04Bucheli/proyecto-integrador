# -*- coding: utf-8 -*-
"""Ensambla el notebook Entregable 1 reproduciendo, celda por celda, la
estructura real del notebook entregado y revisado por el tutor. Las celdas
%%writefile se generan a partir del contenido actual de src/*.py: si cambias
algo ahi, este script lo refleja en el notebook. El resto de celdas (texto
y las celdas de uso contra el dataset real) se generan igual que en el
notebook entregado, para que una regeneracion no invente texto nuevo ni
borre pasos que si se ejecutaron (p.ej. la celda de login de Hugging Face).

Ejecutar desde cualquier directorio:
    python notebooks/build_notebook.py

Regenera notebooks/Fase1_Baselines_y_Auditoria.ipynb a partir de src/.

IMPORTANTE: este script reconstruye el notebook DESDE CERO cada vez que
se ejecuta, es decir, borra cualquier salida de celdas ya ejecutadas. Si ya
entrenaste y evaluaste en Colab y quieres conservar esas salidas, NO vuelvas
a correr este script sobre ese notebook: en vez de eso, aplica el cambio de
codigo a mano, o guarda la version ejecutada con otro nombre antes de
regenerar.
"""
import nbformat as nbf
from pathlib import Path

# Rutas relativas a este archivo, no a un usuario o maquina en particular:
# asi el script funciona igual sin importar donde se haya clonado el repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
OUTPUT_NOTEBOOK = Path(__file__).resolve().parent / "Fase1_Baselines_y_Auditoria.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "colab": {"provenance": []},
    "language_info": {"name": "python"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


def writefile_cell(relpath):
    content = (SRC / relpath).read_text(encoding="utf-8")
    code(f"%%writefile src/{relpath}\n{content}")


# Celdas 0-36: mismo orden y mismo texto que el notebook entregado y
# revisado por el tutor (Entregable 1). Las que van entre ```writefile_cell(``` se arman con el
# contenido vivo de src/ en vez de texto fijo.
md('##Entregable 1 (Baselines, auditoría de fuga de datos y embeddings)\n\n1. Se verifica que el test de\n   sub50_42 no se superpone con el train de full_42 (el split que\n   preentrenó AstroM3-CLIP) por una posible fuga de datos.\n2. Se carga el sub10_42 para el desarrollo local\n3. Se entrenan los tres baselines unimodales: GRU para fotometría,\n   CNN 1D para espectros y MLP para metadatos.\n4. Se hace un experimento de generalización: modelos entrenados en sub10 y evaluados sobre el test de sub50, y luego entrenamiento formal sobre el train de sub50 y evaluado sobre su propio test.\n5. Se extrae el embedding de la capa previa al clasificador de los modelos entrenados en sub50 para la fusión por concatenación.\n\n')
code('!pip install -q datasets torch scikit-learn')
md('No es necesario iniciar sesión en Hugging Face, pero se lo hace para aumentar el límite de solicitudes y la velocidad de descarga.')
code('from huggingface_hub import login\n\nlogin()  # pega aquí tu token cuando te lo pida (no queda guardado en el notebook)')
md('Creamos el paquete src. Cada celda %%writefile escribe uno de los módulos necesarios para el programa')
code("import os\nos.makedirs('src', exist_ok=True)")
writefile_cell('__init__.py')
md('Aquí manejo la configuración central. Definimos SEED = 42 en todo el proyecto para evitar la fuga de datos, según lo que documentan los autores de AstroM3.')
writefile_cell('config.py')
md('Aquí se hace la auditoría en caso de una posible fuga de datos. AstroM3-CLIP fue preentrenado sobre el split train del dataset full con seed 42. Aquí demuestro que ningún objeto del conjunto de prueba (split test de sub50_42) estuvo en ese train.\n\n**Metodología**: cada estrella tiene un vector de metadatos catalográficos de 34\nvalores continuos (posición, magnitudes, periodo, amplitud, etc. — ver\nAstroM3Processed en Hugging Face), en esencia, una huella\ndigital única por objeto. Si el mismo vector de metadatos aparece en full_42/train y en sub50_42/test, ese es el mismo objeto físico')
writefile_cell('audit.py')
md('Aquí empaquetamos los splits de Hugging Face (por requerimiento de PyTorch). Se implementa collate_fn porque la fotometría de cada estrella tiene longitud distinta. Se resuelve aplicando un padding.')
writefile_cell('datasets.py')
md('Aquí se implementan los tres codificadores unimodales de referencia (GRU - fotometría, CNN 1D - espectro, MLP - metadata). Los tres devuelven un embedding de la misma dimensión para poder hacer la fusión por concatenación después')
writefile_cell('models.py')
md('Loop de entrenamiento y evaluación que usan los tres baselines. Reporta exactitud. macroF1, precisión, recall y matriz de confusión')
writefile_cell('engine.py')
md('Aquí se extraen los embeddings de cada modelo ya entrenado. Se guardan para poder concatenarlos después')
writefile_cell('embeddings.py')
code('# reiniciar el intérprete de imports (por si ya se había importado src antes)\nimport importlib, sys\nfor m in list(sys.modules):\n    if m == "src" or m.startswith("src."):\n        del sys.modules[m]\n\nfrom src import config, audit, engine, embeddings as emb_module\nfrom src.datasets import load_subset, inspect_sample, make_dataloader, AstroMultimodalDataset\nfrom src.models import PhotometryGRUEncoder, SpectraCNNEncoder, MetadataMLPEncoder, UnimodalClassifier\nimport torch\n\ntorch.manual_seed(config.SEED)\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\nprint("Dispositivo:", device)')
md('**Auditoría de fuga de datos**\n\nSe verifica que el test de sub50_42 no se superpone con\nel train de full_42.\n')
code('# prueba que la auditoria funciona en sub10\nself_check = audit.run_self_check(subset_name="sub10")\nprint(self_check)\nassert self_check.is_clean, (\n    "El self-check encontró fuga entre su propio "\n    "train y test. chequear el método de auditoria"\n)\n\nprint()\n\n#la auditoría real con full_42\nreport = audit.run_leakage_audit()\nprint(report)\nassert report.is_clean, "Hay fuga de datos"')
md('Carga de sub10_42\n\n')
code('sub10 = load_subset("sub10")\nprint(sub10)\nprint()\ninspect_sample(sub10["train"])')
code('import src.datasets as ds_module\n\n#nombre de las clases en la DB\nreal_class_names = sub10["train"].features["label"].names\nprint("Clases en el dataset:", real_class_names)\nassert real_class_names == config.CLASS_NAMES, "Ajusta config.CLASS_NAMES al orden real antes de seguir."\n\n# dimensión real de metadata (para construir el MLP)\nmetadata_dim = len(sub10["train"][0][ds_module.METADATA_KEY])\nprint("Dimensión de metadatos:", metadata_dim)')
md('**Entrenar los tres baselines sobre sub10**\n\nAquí se entrenan las tres modalidades, empezando por metadatos, luego espectros, y al final fotometría\n')
code('train_loader = make_dataloader(sub10["train"], batch_size=config.BATCH_SIZE, shuffle=True)\nval_loader   = make_dataloader(sub10["validation"], batch_size=config.BATCH_SIZE, shuffle=False)\ntest_loader  = make_dataloader(sub10["test"], batch_size=config.BATCH_SIZE, shuffle=False)')
code('# --- Metadatos (MLP) ---\nmlp_model = UnimodalClassifier(MetadataMLPEncoder(input_dim=metadata_dim), num_classes=config.NUM_CLASSES)\nmlp_result = engine.fit(mlp_model, train_loader, val_loader,\n                         forward_fn=engine.forward_fn_metadata, device=device,\n                         class_names=config.CLASS_NAMES)')
code('# --- Espectros (CNN 1D) ---\ncnn_model = UnimodalClassifier(SpectraCNNEncoder(in_channels=3), num_classes=config.NUM_CLASSES)\ncnn_result = engine.fit(cnn_model, train_loader, val_loader,\n                         forward_fn=engine.forward_fn_spectra, device=device,\n                         class_names=config.CLASS_NAMES)')
code('# --- Fotometría (GRU) ---\ngru_model = UnimodalClassifier(PhotometryGRUEncoder(input_dim=9), num_classes=config.NUM_CLASSES)\ngru_result = engine.fit(gru_model, train_loader, val_loader,\n                         forward_fn=engine.forward_fn_photometry, device=device,\n                         class_names=config.CLASS_NAMES)')
code('# metricas finales de los tres baselines sobre el test de sub10\nfor name, model, forward_fn in [("Metadatos (MLP)", mlp_model, engine.forward_fn_metadata),\n                                 ("Espectros (CNN 1D)", cnn_model, engine.forward_fn_spectra),\n                                 ("Fotometría (GRU)", gru_model, engine.forward_fn_photometry)]:\n    m = engine.evaluate(model, test_loader, device, forward_fn, config.CLASS_NAMES)\n    print(f"\\n=== {name} — test de sub10 ===")\n    print(f"Accuracy: {m[\'accuracy\']:.3f}  |  Macro-F1: {m[\'macro_f1\']:.3f}")\n    print(m["report_text"])')
md('**Experimento de generalización**\n\n1.   modelos entrenados en sub10, evaluados sobre el test de sub50\n2.   entrenamiento sobre el train de sub50, evaluado sobre su propio test')
code('sub50 = load_subset("sub50")\nsub50_test_loader = make_dataloader(sub50["test"], batch_size=config.BATCH_SIZE, shuffle=False)\n\nprint("---- Modelos entrenados en sub10, evaluados en test de sub50 ----")\nfor name, model, forward_fn in [("Metadatos (MLP)", mlp_model, engine.forward_fn_metadata),\n                                 ("Espectros (CNN 1D)", cnn_model, engine.forward_fn_spectra),\n                                 ("Fotometría (GRU)", gru_model, engine.forward_fn_photometry)]:\n    m = engine.evaluate(model, sub50_test_loader, device, forward_fn, config.CLASS_NAMES)\n    print(f"{name}: accuracy={m[\'accuracy\']:.3f}  macro-F1={m[\'macro_f1\']:.3f}")')
code('#entrenamiento sobre sub50 train, validado sobre sub50 validation\nsub50_train_loader = make_dataloader(sub50["train"], batch_size=config.BATCH_SIZE, shuffle=True)\nsub50_val_loader   = make_dataloader(sub50["validation"], batch_size=config.BATCH_SIZE, shuffle=False)\n\nmlp_model_formal = UnimodalClassifier(MetadataMLPEncoder(input_dim=metadata_dim), num_classes=config.NUM_CLASSES)\nmlp_formal_result = engine.fit(mlp_model_formal, sub50_train_loader, sub50_val_loader,\n                                forward_fn=engine.forward_fn_metadata, device=device,\n                                class_names=config.CLASS_NAMES)\n\ncnn_model_formal = UnimodalClassifier(SpectraCNNEncoder(in_channels=3), num_classes=config.NUM_CLASSES)\ncnn_formal_result = engine.fit(cnn_model_formal, sub50_train_loader, sub50_val_loader,\n                                forward_fn=engine.forward_fn_spectra, device=device,\n                                class_names=config.CLASS_NAMES)\n\ngru_model_formal = UnimodalClassifier(PhotometryGRUEncoder(input_dim=9), num_classes=config.NUM_CLASSES)\ngru_formal_result = engine.fit(gru_model_formal, sub50_train_loader, sub50_val_loader,\n                                forward_fn=engine.forward_fn_photometry, device=device,\n                                class_names=config.CLASS_NAMES)')
code('print("---- Entrenamiento en sub50, evaluados en su propio test ----")\nfor name, model, forward_fn in [("Metadatos (MLP)", mlp_model_formal, engine.forward_fn_metadata),\n                                 ("Espectros (CNN 1D)", cnn_model_formal, engine.forward_fn_spectra),\n                                 ("Fotometría (GRU)", gru_model_formal, engine.forward_fn_photometry)]:\n    m = engine.evaluate(model, sub50_test_loader, device, forward_fn, config.CLASS_NAMES)\n    print(f"{name}: accuracy={m[\'accuracy\']:.3f}  macro-F1={m[\'macro_f1\']:.3f}")')
md('**Extracción de embeddings (para la fusión por concatenación)**\n\nAquí se extrae el embedding de la capa previa al clasificador de los modelos\nentrenados en sub50 sobre los 3 splits\n')
code('loaders_ordered = {\n    "train": make_dataloader(sub50["train"], batch_size=config.BATCH_SIZE, shuffle=False),\n    "validation": make_dataloader(sub50["validation"], batch_size=config.BATCH_SIZE, shuffle=False),\n    "test": sub50_test_loader,\n}\n\nos.makedirs("embeddings", exist_ok=True)\n\nfor split_name, loader in loaders_ordered.items():\n    mlp_out = emb_module.extract_embeddings(mlp_model_formal, loader, device, engine.forward_fn_metadata)\n    cnn_out = emb_module.extract_embeddings(cnn_model_formal, loader, device, engine.forward_fn_spectra)\n    gru_out = emb_module.extract_embeddings(gru_model_formal, loader, device, engine.forward_fn_photometry)\n\n    assert (mlp_out["labels"] == cnn_out["labels"]).all() and (mlp_out["labels"] == gru_out["labels"]).all(), \\\n        "labels no coinciden entre modalidades"\n\n    emb_module.save_embeddings(f"embeddings/metadata_{split_name}.npz", mlp_out["embeddings"], mlp_out["labels"])\n    emb_module.save_embeddings(f"embeddings/spectra_{split_name}.npz", cnn_out["embeddings"], cnn_out["labels"])\n    emb_module.save_embeddings(f"embeddings/photometry_{split_name}.npz", gru_out["embeddings"], gru_out["labels"])\n\n    fused = emb_module.concatenate_modalities(\n        gru_out["embeddings"], cnn_out["embeddings"], mlp_out["embeddings"], mlp_out["labels"])\n    emb_module.save_embeddings(f"embeddings/fused_{split_name}.npz", fused["embeddings"], fused["labels"])\n')

nb["cells"] = cells
OUTPUT_NOTEBOOK.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Notebook generado en: {OUTPUT_NOTEBOOK}")
