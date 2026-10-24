# Clasificación multimodal de estrellas variables

Proyecto integrador — David Sebastián Bucheli Cabezas, Ingeniería en Ciencias
de la Computación, USFQ. Tutor: Daniel Riofrío.


## Estructura del repositorio

```
.
├── src/                    # paquete de código (importable como "src")
│   ├── config.py           # semilla, subsets, hiperparámetros
│   ├── audit.py            # auditoría de fuga de datos (huella digital de metadatos)
│   ├── datasets.py         # carga de datos + collate_fn (fotometría de longitud variable)
│   ├── models.py           # GRU (fotometría), CNN 1D (espectros), MLP (metadatos)
│   ├── engine.py           # loop de entrenamiento + evaluación (accuracy, macro-F1, matriz de confusión)
│   ├── embeddings.py       # extracción y concatenación de embeddings
│   ├── runs.py             # (E2) entrenamiento con checkpoints, probabilidades, guardado
│   ├── fusion.py           # (E2) fusión por votación 2-de-3 y análisis de acuerdo
│   ├── explore.py          # (E2) exploración de datos (clases, longitudes, canales)
│   └── reporting.py        # (E2) matrices de confusión, curvas y tablas
├── tests/
│   ├── test_smoke.py       # prueba con datos SINTÉTICOS (no requiere internet)
│   ├── test_fusion.py      # (E2) lógica de la votación, con casos armados a mano
│   └── test_entregable2.py # (E2) flujo completo con datos SINTÉTICOS
├── notebooks/
│   ├── Fase1_Baselines_y_Auditoria.ipynb   # Entregable 1 (ejecutado, con salidas)
│   ├── Entregable2_Sub50_y_Votacion.ipynb  # Entregable 2 (ejecutar con datos reales)
│   └── build_notebook.py                    # regenera SOLO el notebook del Entregable 1
├── embeddings/              # salida del notebook (sección 5); no versionada, ver .gitignore
├── requirements.txt          # dependencias para correr src/, tests/ y el notebook
├── requirements-dev.txt      # dependencia extra solo para build_notebook.py
└── README.md
```

`src/` es un paquete real (tiene `__init__.py`) importado como `from src import ...`
tanto desde `tests/test_smoke.py` como desde el notebook. No hace falta
instalarlo: basta con ejecutar todo desde la raíz del repositorio.

## Instalación

```bash
git clone <url-de-este-repositorio>
cd astro_project
python -m venv .venv && source .venv/bin/activate   # opcional pero recomendado
pip install -r requirements.txt
```

## Cómo reproducir, paso a paso

**1. Prueba de humo (sin internet).** Confirma que el entorno
está bien configurado y que el pipeline completo (collate, los tres
modelos, el loop de entrenamiento, la extracción de embeddings y la lógica
de auditoría de fuga) corre sin errores estructurales, usando datos
sintéticos con las mismas formas documentadas del dataset real:

```bash
python tests/test_smoke.py
```

Esto **no** reemplaza correr contra el dataset real — solo valida que el
código en sí no tiene errores antes de gastar tiempo de cómputo o cuota de
descarga con el dataset real.

**2. Self-check contra el dataset real (requiere internet, descarga solo
`sub10`).** Verifica que el método de auditoría por huella digital detecta
correctamente tanto la ausencia de fuga como una fuga inyectada:

```python
from src import audit
self_check = audit.run_self_check(subset_name="sub10")
print(self_check)
assert self_check.is_clean
```

**3. Auditoría real de fuga de datos (descarga `full`, la más pesada).**
Es la verificación central del proyecto: confirma que el `test` de
`sub50_42` no se superpone con el `train` de `full_42` (el split que
preentrenó `AstroM3-CLIP`):

```python
report = audit.run_leakage_audit()
print(report)
assert report.is_clean
```

**4. Notebook completo (Colab recomendado, o local/infraestructura de la
universidad).** `notebooks/Fase1_Baselines_y_Auditoria.ipynb` corre, en
orden: la auditoría, la carga y exploración de `sub10`, el entrenamiento y
evaluación de los tres baselines sobre `sub10`, el experimento de
generalización `sub10 → sub50`, el entrenamiento formal sobre `sub50`, y la
extracción de embeddings. El propio notebook escribe los archivos de `src/`
en el entorno de ejecución (celdas `%%writefile`), así que no hace falta
subir la carpeta por separado a Colab.


## Semilla: no cambiar sin volver a auditar

Todo el proyecto usa `SEED = 42` (ver `src/config.py`) a propósito: es la
semilla con la que se preentrenó `AstroM3-CLIP`. Mientras se use semilla 42
en todos los subsets, el `test` de cualquier subset queda garantizado fuera
del `train` que vio el preentrenamiento. Si en algún momento se evalúa la
fusión por transferencia de aprendizaje con otra semilla, hay que volver a
correr `audit.run_leakage_audit()` con esa semilla antes de reportar
resultados.

Nota sobre la auditoría: como los subsets de AstroM3Processed se muestrean
de los splits de `full`, la intersección vacía entre `sub50/test` y
`full/train` está prácticamente garantizada por construcción del dataset,
siempre que se mantenga la semilla 42. La auditoría es, por tanto, un
control necesario 

