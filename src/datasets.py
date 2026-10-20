# -*- coding: utf-8 -*-

from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from . import config

PHOTOMETRY_KEY = "photometry"
SPECTRA_KEY = "spectra"
METADATA_KEY = "metadata"
LABEL_KEY = "label"


#verifica formato del dataset
def inspect_sample(hf_split) -> None:
    example = hf_split[0]
    print("Claves disponibles:", list(example.keys()))
    for key, value in example.items():
        try:
            arr = torch.as_tensor(value)
            print(f"  {key}: tensor de forma {tuple(arr.shape)}, dtype origen {type(value)}")
        except (TypeError, ValueError):
            print(f"  {key}: {value!r} (no numérico / no convertible directo)")


#envuelve un split de la BD AstroM3Processed
class AstroMultimodalDataset(Dataset):

    def __init__(self, hf_split, photometry_key=PHOTOMETRY_KEY,
                 spectra_key=SPECTRA_KEY, metadata_key=METADATA_KEY,
                 label_key=LABEL_KEY):
        self.data = hf_split
        self.photometry_key = photometry_key
        self.spectra_key = spectra_key
        self.metadata_key = metadata_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data[idx]

        photometry = torch.as_tensor(row[self.photometry_key], dtype=torch.float32)
        #forma (T, canales)
        if photometry.ndim == 1:
            photometry = photometry.unsqueeze(-1)

        spectra = torch.as_tensor(row[self.spectra_key], dtype=torch.float32)

        metadata = torch.as_tensor(row[self.metadata_key], dtype=torch.float32).flatten()

        label = int(row[self.label_key])

        return {
            "photometry": photometry,
            "spectra": spectra,
            "metadata": metadata,
            "label": label,
        }


#arma un lote rellenando la fotometria a la longitud máx. del batch. Devuelvo la longitud real para poder empaquetar la secuencia en el GRU.
def collate_multimodal(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

    photometry_list = [b["photometry"] for b in batch]
    lengths = torch.tensor([p.shape[0] for p in photometry_list], dtype=torch.long)
    photometry_padded = pad_sequence(photometry_list, batch_first=True)  # (B, T_max, C)

    spectra = torch.stack([b["spectra"] for b in batch])      # (B, 3, L)
    metadata = torch.stack([b["metadata"] for b in batch])    # (B, D)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    return {
        "photometry": photometry_padded,
        "photometry_lengths": lengths,
        "spectra": spectra,
        "metadata": metadata,
        "label": labels,
    }


def make_dataloader(hf_split, batch_size: int = config.BATCH_SIZE,
                     shuffle: bool = True, **dataset_kwargs) -> DataLoader:
    ds = AstroMultimodalDataset(hf_split, **dataset_kwargs)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_multimodal,
    )

#carga un subset con la semilla definida
def load_subset(subset_name: str, seed: int = config.SEED,
                 dataset_name: str = config.DATASET_NAME):
    from datasets import load_dataset
    return load_dataset(dataset_name, name=f"{subset_name}_{seed}")
