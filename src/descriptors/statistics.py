# src/descriptors/statistics.py
"""
Descriptores estadísticos de primer orden.
"""

import numpy as np
from typing import Dict
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy


def first_order_stats(img: np.ndarray) -> Dict[str, float]:
    """
    Calcula estadísticas de primer orden de una imagen.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises.
    
    Returns
    -------
    dict
        Diccionario con: mean, var, skew, kurtosis, entropy.
    """
    img_f = img.astype(np.float32).ravel()
    
    feats = {
        "mean": float(img_f.mean()),
        "var": float(img_f.var()),
        "skew": float(skew(img_f)),
        "kurtosis": float(kurtosis(img_f)),
        "entropy": float(shannon_entropy(img))
    }
    
    return feats


def first_order_stats_vector(img: np.ndarray) -> np.ndarray:
    """
    Calcula estadísticas de primer orden como vector.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises.
    
    Returns
    -------
    np.ndarray
        Vector de características [mean, var, skew, kurtosis, entropy].
    """
    stats = first_order_stats(img)
    return np.array([
        stats["mean"],
        stats["var"],
        stats["skew"],
        stats["kurtosis"],
        stats["entropy"]
    ], dtype=np.float32)

