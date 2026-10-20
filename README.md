# Clasificación multimodal de estrellas variables

Proyecto integrador — David Sebastián Bucheli Cabezas, Ingeniería en Ciencias
de la Computación, USFQ. Tutor: Daniel Riofrío.

Código de la Fase 1: auditoría de integridad de datos, los tres modelos
unimodales de referencia (fotometría, espectros, metadatos), y la extracción
de embeddings para la fusión por concatenación. Dataset:
[`AstroMLCore/AstroM3Processed`](https://huggingface.co/datasets/AstroMLCore/AstroM3Processed)
(Hugging Face).

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

**1. Prueba de humo (sin internet, ~1 minuto).** Confirma que el entorno
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

> ⚠️ El notebook incluido en este repositorio (`notebooks/Fase1_Baselines_y_Auditoria.ipynb`)
> ya se corrió completo hasta la sección 4 (auditoría + los tres baselines
> sobre `sub10` y `sub50`) y se guardó **con las salidas visibles**
> (historial de épocas, métricas de test, reporte por clase). La sección 5
> (extracción de embeddings) todavía no se ha ejecutado — ver "Estado
> actual" abajo. Si vuelves a correr el notebook completo, revisa que esa
> sección también termine y quede con sus salidas antes de dar por cerrado
> este entregable.

## Estado actual (honesto, a la fecha de este commit)

- ✅ Auditoría de fuga de datos: implementada y **ejecutada contra datos
  reales**, con hallazgo real documentado (AstroM3Processed no tiene columna
  de ID; se audita por huella digital de metadatos). Resultado: 0
  intersecciones entre `full_42/train` (17063 objetos) y `sub50_42/test`
  (1111 objetos) — sin fuga detectada.
- ✅ Los tres baselines unimodales (GRU-fotometría, CNN 1D-espectros,
  MLP-metadatos) están **entrenados y evaluados sobre datos reales**, con
  salidas guardadas en el notebook:
  - Sobre `sub10` (test, 220 objetos): GRU 90.5 % accuracy / 0.904 macro-F1,
    MLP 75.0 % / 0.734, CNN 1D 49.1 % / 0.453.
  - Generalización `sub10 → test de sub50` (modelos entrenados en `sub10`,
    nunca vieron `sub50`): GRU 66.3 % / 0.663, MLP 68.3 % / 0.630,
    CNN 1D 55.0 % / 0.458.
  - Entrenamiento formal sobre `sub50` (evaluado en su propio test, 1111
    objetos): GRU 88.6 % / 0.897, MLP 82.0 % / 0.788, CNN 1D 60.9 % / 0.478.
  - La fotometría (GRU) es, con estos datos, la modalidad más fuerte de las
    tres; el espectro (CNN 1D) es sistemáticamente la más débil.
- ⏳ Extracción de embeddings (sección 5 del notebook, insumo para la fusión
  por concatenación): el código está listo y validado **estructuralmente**
  con `tests/test_smoke.py` (datos sintéticos), pero la celda todavía no se
  ha corrido contra los modelos ya entrenados sobre `sub50`.
- ⏳ Fusión por votación, concatenación y transferencia de aprendizaje:
  siguiente fase del proyecto (Entregable 2).

## Entregable 2: sub50 reentrenado y votación 2-de-3

Código nuevo, en módulos aparte (no modifica los módulos del Entregable 1):
`runs.py`, `fusion.py`, `explore.py`, `reporting.py`. El notebook
`notebooks/Entregable2_Sub50_y_Votacion.ipynb` hace, en orden: exploración de
`sub50`, reentrenamiento de los tres modelos con el mismo protocolo del
Entregable 1 (guardando checkpoints, historiales y probabilidades), evaluación
con matrices de confusión, votación 2-de-3 y extracción de embeddings.

```bash
python tests/test_fusion.py          # lógica de la votación (sin torch ni internet)
python tests/test_entregable2.py     # flujo completo con datos sintéticos
```

Estas pruebas validan que el código corre; **no producen resultados**. Los
resultados del Entregable 2 salen de ejecutar el notebook con el dataset real y
quedan en `outputs_e2/` (`results.json`, historiales, figuras y predicciones).
Checkpoints (`*.pt`) y embeddings (`*.npz`) se generan pero no se versionan.

Decisiones de protocolo: modelo principal = última época (igual que el
Entregable 1); en la votación, la regla de desempate cuando los tres modelos
difieren se elige con validación, nunca con test.

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
control de cordura necesario y bien ejecutado, no un hallazgo metodológico
nuevo — así se reporta también en el documento final.

## Regenerar el notebook desde `src/`

Si cambias algo en `src/`, el notebook no se actualiza solo. Para
reconstruirlo:

```bash
pip install -r requirements-dev.txt
python notebooks/build_notebook.py
```

⚠️ Esto **reescribe el notebook desde cero** y borra cualquier salida de
celdas ya ejecutadas. Si ya entrenaste en Colab y quieres conservar esas
salidas, no lo vuelvas a correr sobre ese mismo archivo: aplica el cambio de
código a mano, o guarda la versión ejecutada con otro nombre antes de
regenerar. (Esto es justamente lo que causó que una versión anterior del
notebook se entregara sin salidas de entrenamiento.)

## Próximos pasos

Con `embeddings/fused_train.npz`, `fused_validation.npz` y `fused_test.npz`
generados, la siguiente fase es entrenar el clasificador de la fusión por
concatenación sobre esos vectores, implementar la fusión por votación
mayoritaria, y evaluar la transferencia de aprendizaje desde
`AstroM3-CLIP` — todo sobre `sub50`, usando la infraestructura de cómputo de
la universidad.
