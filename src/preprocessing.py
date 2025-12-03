# src/preprocessing.py
"""
Funciones de preprocesamiento de imágenes.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# Tamaño por defecto para redimensionamiento
IMG_SIZE = (256, 256)


def read_and_preprocess(
    path: str,
    img_size: Tuple[int, int] = IMG_SIZE,
    apply_clahe: bool = True
) -> np.ndarray:
    """
    Lee y preprocesa una imagen en escala de grises.
    
    Parameters
    ----------
    path : str
        Ruta a la imagen.
    img_size : Tuple[int, int]
        Tamaño al que redimensionar la imagen (ancho, alto).
    apply_clahe : bool
        Si True, aplica ecualización adaptativa de histograma (CLAHE).
    
    Returns
    -------
    np.ndarray
        Imagen preprocesada normalizada en rango [0, 1].
    
    Raises
    ------
    ValueError
        Si no se puede leer la imagen.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer: {path}")

    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    return img


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Aplica ecualización adaptativa de histograma (CLAHE) a una imagen.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises (uint8).
    clip_limit : float
        Límite de contraste para CLAHE.
    tile_grid_size : Tuple[int, int]
        Tamaño de la cuadrícula de tiles.
    
    Returns
    -------
    np.ndarray
        Imagen con CLAHE aplicado.
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def resize_image(
    img: np.ndarray,
    size: Tuple[int, int] = IMG_SIZE,
    interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    """
    Redimensiona una imagen al tamaño especificado.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada.
    size : Tuple[int, int]
        Tamaño objetivo (ancho, alto).
    interpolation : int
        Método de interpolación de OpenCV.
    
    Returns
    -------
    np.ndarray
        Imagen redimensionada.
    """
    return cv2.resize(img, size, interpolation=interpolation)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normaliza una imagen al rango [0, 1].
    
    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada (uint8 o float).
    
    Returns
    -------
    np.ndarray
        Imagen normalizada en rango [0, 1].
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img


def get_image_sizes(paths: list) -> list:
    """
    Obtiene los tamaños de una lista de imágenes.
    
    Parameters
    ----------
    paths : list
        Lista de rutas a las imágenes.
    
    Returns
    -------
    list
        Lista de tuplas (ancho, alto).
    """
    sizes = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            sizes.append((w, h))
    return sizes

