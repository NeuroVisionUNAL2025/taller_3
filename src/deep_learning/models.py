# src/deep_learning/models.py
"""
Arquitecturas de redes neuronales para clasificación de imágenes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Red neuronal convolucional simple para clasificación binaria.
    
    Arquitectura:
    - 3 bloques convolucionales con ReLU y MaxPool
    - Global Average Pooling
    - Clasificador con dropout
    
    Parameters
    ----------
    in_channels : int
        Número de canales de entrada (default: 3 para RGB).
    """
    
    def __init__(self, in_channels: int = 3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada (batch, channels, height, width).
        
        Returns
        -------
        torch.Tensor
            Probabilidades de salida (batch, 1).
        """
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x)
    
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass retornando logits (sin sigmoid).
        
        Útil cuando se usa BCEWithLogitsLoss.
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada.
        
        Returns
        -------
        torch.Tensor
            Logits de salida.
        """
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_device() -> torch.device:
    """
    Obtiene el dispositivo disponible (GPU si está disponible, sino CPU).
    
    Returns
    -------
    torch.device
        Dispositivo de PyTorch.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model: nn.Module) -> int:
    """
    Cuenta el número de parámetros entrenables en un modelo.
    
    Parameters
    ----------
    model : nn.Module
        Modelo de PyTorch.
    
    Returns
    -------
    int
        Número de parámetros entrenables.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

