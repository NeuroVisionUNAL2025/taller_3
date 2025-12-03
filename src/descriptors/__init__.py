# src/descriptors/__init__.py
"""
Descriptores de características para imágenes.
- shape: HOG, Hu moments, contornos, Fourier
- texture: LBP, GLCM, Gabor
- statistics: estadísticas de primer orden
"""

from .shape import (
    descriptor_hog,
    descriptor_hu,
    descriptor_contorno,
    descriptor_fourier,
    segmentar
)
from .texture import (
    descriptor_lbp,
    descriptor_glcm,
    descriptor_gabor
)
from .statistics import first_order_stats

__all__ = [
    # Shape
    'descriptor_hog',
    'descriptor_hu',
    'descriptor_contorno',
    'descriptor_fourier',
    'segmentar',
    # Texture
    'descriptor_lbp',
    'descriptor_glcm',
    'descriptor_gabor',
    # Statistics
    'first_order_stats',
]

