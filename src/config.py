# -*- coding: utf-8 -*-

SEED = 42
DATASET_NAME = "AstroMLCore/AstroM3Processed"

SUBSET_DEV = "sub10"      # desarrollo local
SUBSET_FORMAL = "sub50"   # entrenamiento formal
SUBSET_FULL = "full"      # para la detección de fuga de datos

NUM_CLASSES = 10
CLASS_NAMES = ["DSCT", "EA", "EB", "EW", "HADS", "M", "ROT", "RRAB", "RRC", "SR"]

#modelo preentrenado para la fusión por transfer learning contrastivo
CLIP_MODEL_NAME = "AstroMLCore/AstroM3-CLIP"

#misma dimensión de embedding para los tres codificadores unimodales (más conveniente para la fusión por concatenación)
EMBEDDING_DIM = 64

#hiperparámetros de entrenamiento
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
RANDOM_STATE_TORCH = SEED
