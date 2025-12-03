# src/deep_learning/dataset.py
"""
Dataset de PyTorch para carga de imágenes de rayos X.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple, Optional


def get_transform(img_size: int = 224) -> T.Compose:
    """
    Retorna las transformaciones estándar para las imágenes.
    
    Parameters
    ----------
    img_size : int
        Tamaño de la imagen de salida.
    
    Returns
    -------
    T.Compose
        Composición de transformaciones.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class ChestXrayDataset(Dataset):
    """
    Dataset de PyTorch para imágenes de rayos X de tórax.
    
    Parameters
    ----------
    paths : List[str]
        Lista de rutas a las imágenes.
    labels : List[str]
        Lista de etiquetas ('NORMAL', 'PNEUMONIA').
    transform : callable, optional
        Transformaciones a aplicar a las imágenes.
    label_map : dict, optional
        Mapeo de etiquetas a valores numéricos.
    """
    
    def __init__(
        self,
        paths: List[str],
        labels: List[str],
        transform: Optional[T.Compose] = None,
        label_map: Optional[dict] = None
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform if transform is not None else get_transform()
        self.label_map = label_map if label_map is not None else {
            "NORMAL": 0.0,
            "PNEUMONIA": 1.0
        }

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene un elemento del dataset.
        
        Parameters
        ----------
        idx : int
            Índice del elemento.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (imagen, etiqueta)
        """
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        
        label_str = self.labels[idx]
        label = self.label_map[label_str]

        return img, torch.tensor(label, dtype=torch.float32)


def create_dataloaders(
    paths_train: List[str],
    labels_train: List[str],
    paths_val: List[str],
    labels_val: List[str],
    paths_test: List[str],
    labels_test: List[str],
    batch_size: int = 32,
    num_workers: int = 0,
    transform: Optional[T.Compose] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para entrenamiento, validación y prueba.
    
    Parameters
    ----------
    paths_train, labels_train : List[str]
        Datos de entrenamiento.
    paths_val, labels_val : List[str]
        Datos de validación.
    paths_test, labels_test : List[str]
        Datos de prueba.
    batch_size : int
        Tamaño del batch.
    num_workers : int
        Número de workers para carga de datos.
    transform : T.Compose, optional
        Transformaciones a aplicar.
    
    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    train_dataset = ChestXrayDataset(paths_train, labels_train, transform)
    val_dataset = ChestXrayDataset(paths_val, labels_val, transform)
    test_dataset = ChestXrayDataset(paths_test, labels_test, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

