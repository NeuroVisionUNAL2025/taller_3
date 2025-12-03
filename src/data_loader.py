# src/data_loader.py
"""
Funciones para carga de datos y rutas de imágenes.
"""

import os
from glob import glob
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_image_paths(data_dir: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Carga las rutas de imágenes y sus etiquetas desde un directorio estructurado.
    
    Espera una estructura de carpetas:
        data_dir/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
    
    Parameters
    ----------
    data_dir : str
        Ruta al directorio raíz del dataset.
    
    Returns
    -------
    paths : List[str]
        Lista de rutas a las imágenes.
    labels : List[Tuple[str, str]]
        Lista de tuplas (split, clase) donde split es 'train', 'val' o 'test'
        y clase es 'NORMAL' o 'PNEUMONIA'.
    """
    classes = ['NORMAL', 'PNEUMONIA']
    paths = []
    labels = []

    for split in ['train', 'val', 'test']:
        for cls in classes:
            folder = os.path.join(data_dir, split, cls)
            if not os.path.exists(folder):
                continue

            for p in (glob(os.path.join(folder, '*.jpeg')) +
                      glob(os.path.join(folder, '*.jpg')) +
                      glob(os.path.join(folder, '*.png'))):
                paths.append(p)
                labels.append((split, cls))
    
    return paths, labels


def split_by_set(
    paths: List[str],
    labels: List[Tuple[str, str]]
) -> Tuple[
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str]]
]:
    """
    Separa las rutas y etiquetas en conjuntos train, val y test.
    
    Parameters
    ----------
    paths : List[str]
        Lista de rutas a las imágenes.
    labels : List[Tuple[str, str]]
        Lista de tuplas (split, clase).
    
    Returns
    -------
    train : Tuple[List[str], List[str]]
        (paths_train, labels_train)
    val : Tuple[List[str], List[str]]
        (paths_val, labels_val)
    test : Tuple[List[str], List[str]]
        (paths_test, labels_test)
    """
    paths_train = [p for p, (split, cls) in zip(paths, labels) if split == 'train']
    labels_train = [cls for split, cls in labels if split == 'train']

    paths_val = [p for p, (split, cls) in zip(paths, labels) if split == 'val']
    labels_val = [cls for split, cls in labels if split == 'val']

    paths_test = [p for p, (split, cls) in zip(paths, labels) if split == 'test']
    labels_test = [cls for split, cls in labels if split == 'test']

    return (paths_train, labels_train), (paths_val, labels_val), (paths_test, labels_test)


def labels_to_numeric(
    labels: List[str],
    label_map: dict = None
) -> np.ndarray:
    """
    Convierte etiquetas de texto a valores numéricos.
    
    Parameters
    ----------
    labels : List[str]
        Lista de etiquetas de texto ('NORMAL', 'PNEUMONIA').
    label_map : dict, optional
        Diccionario de mapeo. Por defecto: {'NORMAL': 0, 'PNEUMONIA': 1}
    
    Returns
    -------
    np.ndarray
        Array de etiquetas numéricas.
    """
    if label_map is None:
        label_map = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    return np.array([label_map[l] for l in labels], dtype=np.int32)


def get_class_distribution(labels: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Obtiene la distribución de clases en el dataset.
    
    Parameters
    ----------
    labels : List[Tuple[str, str]]
        Lista de tuplas (split, clase).
    
    Returns
    -------
    pd.DataFrame
        DataFrame con la distribución de clases.
    """
    df = pd.DataFrame({'label': labels})
    return df['label'].value_counts()

