# src/features.py
"""
Funciones para extracción combinada de características y construcción de matrices.
"""

import numpy as np
from typing import List, Callable
from tqdm import tqdm

from .preprocessing import read_and_preprocess
from .descriptors import (
    descriptor_hog,
    descriptor_hu,
    descriptor_fourier,
    descriptor_lbp,
    descriptor_glcm,
    descriptor_gabor
)


def extract_features_all(
    img: np.ndarray,
    hog_fn: Callable = None,
    hu_fn: Callable = None,
    fourier_fn: Callable = None,
    lbp_fn: Callable = None,
    glcm_fn: Callable = None,
    gabor_fn: Callable = None
) -> np.ndarray:
    """
    Extrae todas las características de una imagen combinando múltiples descriptores.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    hog_fn : Callable, optional
        Función para calcular HOG. Por defecto usa descriptor_hog.
    hu_fn : Callable, optional
        Función para calcular momentos de Hu. Por defecto usa descriptor_hu.
    fourier_fn : Callable, optional
        Función para calcular Fourier. Por defecto usa descriptor_fourier.
    lbp_fn : Callable, optional
        Función para calcular LBP. Por defecto usa descriptor_lbp.
    glcm_fn : Callable, optional
        Función para calcular GLCM. Por defecto usa descriptor_glcm.
    gabor_fn : Callable, optional
        Función para calcular Gabor. Por defecto usa descriptor_gabor.
    
    Returns
    -------
    np.ndarray
        Vector de características concatenado.
    """
    # Usar funciones por defecto si no se proporcionan
    if hog_fn is None:
        hog_fn = descriptor_hog
    if hu_fn is None:
        hu_fn = descriptor_hu
    if fourier_fn is None:
        fourier_fn = descriptor_fourier
    if lbp_fn is None:
        lbp_fn = descriptor_lbp
    if glcm_fn is None:
        glcm_fn = descriptor_glcm
    if gabor_fn is None:
        gabor_fn = descriptor_gabor
    
    img_u8 = (img * 255).astype("uint8")
    feats = []

    # HOG
    hog_vec, _ = hog_fn(img)
    feats.extend(hog_vec.flatten().tolist())

    # Hu moments
    hu = hu_fn(img)
    feats.extend(hu.tolist())

    # Fourier
    coeff = fourier_fn(img)
    feats.extend(np.abs(coeff).tolist())

    # LBP
    _, lbp_hist = lbp_fn(img, P=8, R=2)
    feats.extend(lbp_hist.flatten().tolist())

    # GLCM
    glcm_feats = glcm_fn(img_u8)
    feats.extend(glcm_feats.tolist())

    # Gabor
    gabor_feats = gabor_fn(img)
    feats.extend(np.array(gabor_feats, dtype=np.float32).flatten().tolist())

    return np.array(feats, dtype=np.float32)


def build_feature_matrix(
    paths: List[str],
    hog_fn: Callable = None,
    hu_fn: Callable = None,
    fourier_fn: Callable = None,
    lbp_fn: Callable = None,
    glcm_fn: Callable = None,
    gabor_fn: Callable = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Construye una matriz de características para un conjunto de imágenes.
    
    Parameters
    ----------
    paths : List[str]
        Lista de rutas a las imágenes.
    hog_fn : Callable, optional
        Función para calcular HOG.
    hu_fn : Callable, optional
        Función para calcular momentos de Hu.
    fourier_fn : Callable, optional
        Función para calcular Fourier.
    lbp_fn : Callable, optional
        Función para calcular LBP.
    glcm_fn : Callable, optional
        Función para calcular GLCM.
    gabor_fn : Callable, optional
        Función para calcular Gabor.
    show_progress : bool
        Si True, muestra barra de progreso.
    
    Returns
    -------
    np.ndarray
        Matriz de características (n_samples, n_features).
    """
    X = []
    
    iterator = tqdm(paths) if show_progress else paths

    for p in iterator:
        try:
            img = read_and_preprocess(p)
            if img is None:
                print(f"Imagen inválida: {p}")
                continue

            features = extract_features_all(
                img,
                hog_fn, hu_fn, fourier_fn,
                lbp_fn, glcm_fn, gabor_fn
            )

            if features.ndim != 1:
                print(f"Vector con shape raro en {p}: {features.shape}")
                continue

            X.append(features)

        except Exception as e:
            print(f"Error al procesar {p}: {e}")

    return np.array(X, dtype=np.float32)


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray = None,
    X_test: np.ndarray = None
):
    """
    Normaliza las características usando StandardScaler.
    
    Parameters
    ----------
    X_train : np.ndarray
        Matriz de características de entrenamiento.
    X_val : np.ndarray, optional
        Matriz de características de validación.
    X_test : np.ndarray, optional
        Matriz de características de prueba.
    
    Returns
    -------
    tuple
        (X_train_norm, X_val_norm, X_test_norm, scaler)
        Los valores None se retornan como None.
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    
    X_val_norm = scaler.transform(X_val) if X_val is not None else None
    X_test_norm = scaler.transform(X_test) if X_test is not None else None
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


def apply_pca(
    X_train: np.ndarray,
    X_val: np.ndarray = None,
    X_test: np.ndarray = None,
    variance_ratio: float = 0.95
):
    """
    Aplica PCA para reducción de dimensionalidad.
    
    Parameters
    ----------
    X_train : np.ndarray
        Matriz de características de entrenamiento.
    X_val : np.ndarray, optional
        Matriz de características de validación.
    X_test : np.ndarray, optional
        Matriz de características de prueba.
    variance_ratio : float
        Porcentaje de varianza a retener (0-1).
    
    Returns
    -------
    tuple
        (X_train_pca, X_val_pca, X_test_pca, pca)
        Los valores None se retornan como None.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(variance_ratio)
    X_train_pca = pca.fit_transform(X_train)
    
    X_val_pca = pca.transform(X_val) if X_val is not None else None
    X_test_pca = pca.transform(X_test) if X_test is not None else None
    
    return X_train_pca, X_val_pca, X_test_pca, pca

