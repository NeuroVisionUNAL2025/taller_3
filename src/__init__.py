# src/__init__.py
"""
Módulos para clasificación de imágenes médicas (Rayos X de tórax).
Comparación de técnicas clásicas de extracción de características vs Deep Learning.

Módulos disponibles:
- data_loader: carga de datos y rutas
- preprocessing: preprocesamiento de imágenes (CLAHE, resize)
- descriptors: descriptores de características (HOG, Hu, LBP, GLCM, Gabor)
- features: extracción combinada de características
- classical_models: modelos clásicos de ML (SVM, RF, kNN, LogReg)
- deep_learning: redes neuronales convolucionales
- evaluation: métricas de evaluación
- visualization: utilidades de visualización
"""

from . import data_loader
from . import preprocessing
from . import descriptors
from . import features
from . import classical_models
from . import deep_learning
from . import evaluation
from . import visualization

__all__ = [
    'data_loader',
    'preprocessing',
    'descriptors',
    'features',
    'classical_models',
    'deep_learning',
    'evaluation',
    'visualization',
]

__version__ = '0.1.0'
