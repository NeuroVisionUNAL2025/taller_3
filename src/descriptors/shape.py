# src/descriptors/shape.py
"""
Descriptores de forma: HOG, Hu moments, contornos y Fourier.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from skimage.feature import hog


def descriptor_hog(
    img: np.ndarray,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    orientations: int = 9,
    cells_per_block: Tuple[int, int] = (2, 2),
    visualize: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calcula el descriptor HOG (Histogram of Oriented Gradients) de una imagen.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    pixels_per_cell : Tuple[int, int]
        Tamaño de celda en píxeles.
    orientations : int
        Número de bins de orientación.
    cells_per_block : Tuple[int, int]
        Número de celdas por bloque.
    visualize : bool
        Si True, retorna también la imagen de visualización.
    
    Returns
    -------
    hog_vec : np.ndarray
        Vector de características HOG.
    hog_img : np.ndarray or None
        Imagen de visualización HOG (si visualize=True).
    """
    img_u8 = (img * 255).astype("uint8")
    
    result = hog(
        img_u8,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize
    )
    
    if visualize:
        hog_vec, hog_img = result
        return hog_vec, hog_img
    else:
        return result, None


def descriptor_hu(img: np.ndarray) -> np.ndarray:
    """
    Calcula los 7 momentos de Hu de una imagen.
    
    Los momentos de Hu son invariantes a traslación, escala y rotación.
    Se retorna la transformación logarítmica para mejor manejo numérico.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    
    Returns
    -------
    np.ndarray
        Vector de 7 momentos de Hu (transformación logarítmica).
    """
    img_u8 = (img * 255).astype("uint8")
    m = cv2.moments(img_u8)
    hu = cv2.HuMoments(m).flatten()
    
    # Transformación logarítmica para mejor manejo numérico
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log


def segmentar(img: np.ndarray) -> np.ndarray:
    """
    Segmenta una imagen usando umbralización de Otsu.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    
    Returns
    -------
    np.ndarray
        Máscara binaria (uint8).
    """
    img_u8 = (img * 255).astype("uint8")
    _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def descriptor_contorno(img: np.ndarray) -> Dict[str, float]:
    """
    Calcula descriptores geométricos del contorno principal de una imagen.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    
    Returns
    -------
    dict
        Diccionario con: area, perimetro, circularidad, excentricidad.
    """
    mask = segmentar(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return {
            "area": 0,
            "perimetro": 0,
            "circularidad": 0,
            "excentricidad": 0
        }

    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    circularidad = 4 * np.pi * area / (perimetro**2 + 1e-12)

    # Excentricidad vía elipse ajustada
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        if ma > 0:
            excentricidad = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)
        else:
            excentricidad = np.nan
    else:
        excentricidad = np.nan

    return {
        "area": area,
        "perimetro": perimetro,
        "circularidad": circularidad,
        "excentricidad": excentricidad
    }


def descriptor_fourier(img: np.ndarray, n_coeff: int = 20) -> np.ndarray:
    """
    Calcula los descriptores de Fourier del contorno principal.
    
    Los descriptores de Fourier capturan la forma del contorno
    y son invariantes a escala cuando se normalizan.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    n_coeff : int
        Número de coeficientes de Fourier a retornar.
    
    Returns
    -------
    np.ndarray
        Vector de coeficientes de Fourier normalizados (complejos).
    """
    mask = segmentar(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(cnts) == 0:
        return np.zeros(n_coeff, dtype=np.complex128)

    cnt = max(cnts, key=len).squeeze()
    
    if cnt.ndim < 2:
        return np.zeros(n_coeff, dtype=np.complex128)
    
    contorno = cnt[:, 0] + 1j * cnt[:, 1]

    fd = np.fft.fft(contorno)

    # Normalización (invarianza a escala)
    fd_norm = fd / (np.abs(fd[1]) + 1e-9)

    return fd_norm[:n_coeff]


def test_hu_invariances(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Prueba las invariancias de los momentos de Hu ante transformaciones.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    
    Returns
    -------
    dict
        Diccionario con los momentos de Hu para cada transformación.
    """
    results = {"original": descriptor_hu(img)}
    
    # Traslación
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    shift = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    results["traslacion"] = descriptor_hu(shift)

    # Rotación 90°
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 90, 1)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    results["rotacion_90"] = descriptor_hu(rot)

    # Escalado 0.5x
    scale = cv2.resize(img, None, fx=0.5, fy=0.5)
    results["escalado_05x"] = descriptor_hu(scale)

    return results

