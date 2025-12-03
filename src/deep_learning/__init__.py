# src/deep_learning/__init__.py
"""
Módulos de Deep Learning para clasificación de imágenes.
- models: arquitecturas de redes neuronales (SimpleCNN)
- dataset: datasets de PyTorch para carga de datos
- training: funciones de entrenamiento y evaluación
"""

from .models import SimpleCNN
from .dataset import ChestXrayDataset, get_transform
from .training import train_cnn, evaluate_cnn

__all__ = [
    'SimpleCNN',
    'ChestXrayDataset',
    'get_transform',
    'train_cnn',
    'evaluate_cnn',
]

