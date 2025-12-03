# src/descriptors/texture.py
"""
Descriptores de textura: LBP, GLCM y Gabor.
"""

import numpy as np
from typing import Tuple, List
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor


def descriptor_lbp(
    img: np.ndarray,
    P: int = 8,
    R: float = 1,
    method: str = 'uniform'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el descriptor LBP (Local Binary Pattern) de una imagen.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    P : int
        Número de puntos vecinos.
    R : float
        Radio del círculo de vecinos.
    method : str
        Método de LBP ('uniform', 'default', 'ror', 'var').
    
    Returns
    -------
    lbp : np.ndarray
        Imagen LBP.
    hist : np.ndarray
        Histograma normalizado de patrones LBP.
    """
    lbp = local_binary_pattern(img, P, R, method=method)
    
    # Para método 'uniform', el número de bins es P + 2
    n_bins = P + 2 if method == 'uniform' else 2**P
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1), density=True)
    
    return lbp, hist


def descriptor_glcm(
    img: np.ndarray,
    distances: List[int] = None,
    angles: List[float] = None,
    levels: int = 256,
    props: List[str] = None
) -> np.ndarray:
    """
    Calcula descriptores GLCM (Gray Level Co-occurrence Matrix).
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1] o uint8.
    distances : List[int]
        Lista de distancias para GLCM. Por defecto: [1]
    angles : List[float]
        Lista de ángulos en radianes. Por defecto: [0]
    levels : int
        Número de niveles de gris.
    props : List[str]
        Propiedades a calcular. Por defecto: ['contrast', 'correlation', 'energy', 'homogeneity']
    
    Returns
    -------
    np.ndarray
        Vector de características GLCM.
    """
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0]
    if props is None:
        props = ['contrast', 'correlation', 'energy', 'homogeneity']
    
    # Convertir a uint8 si es necesario
    if img.dtype != np.uint8:
        img_u8 = (img * 255).astype('uint8')
    else:
        img_u8 = img

    glcm = graycomatrix(
        img_u8,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    feats = []
    for p in props:
        vals = graycoprops(glcm, p)
        feats.append(vals.mean())

    return np.array(feats)


def descriptor_gabor(
    img: np.ndarray,
    freqs: List[float] = None,
    thetas: List[float] = None
) -> np.ndarray:
    """
    Calcula descriptores basados en filtros de Gabor.
    
    Los filtros de Gabor capturan información de textura a diferentes
    frecuencias y orientaciones.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    freqs : List[float]
        Lista de frecuencias. Por defecto: [0.1, 0.2, 0.3]
    thetas : List[float]
        Lista de ángulos en radianes. Por defecto: [0, π/4, π/2]
    
    Returns
    -------
    np.ndarray
        Vector de características (media y desviación estándar por cada combinación).
    """
    if freqs is None:
        freqs = [0.1, 0.2, 0.3]
    if thetas is None:
        thetas = [0, np.pi/4, np.pi/2]
    
    feats = []
    for f in freqs:
        for t in thetas:
            real, imag = gabor(img, frequency=f, theta=t)
            feats.append(real.mean())
            feats.append(real.std())
    
    return np.array(feats)

