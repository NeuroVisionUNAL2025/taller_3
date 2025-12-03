# src/deep_learning/training.py
"""
Funciones de entrenamiento y evaluación para modelos de Deep Learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


def get_class_weights(
    labels: List[str],
    device: torch.device = None
) -> torch.Tensor:
    """
    Calcula los pesos de clase para manejo de desbalance.
    
    Parameters
    ----------
    labels : List[str]
        Lista de etiquetas de texto.
    device : torch.device, optional
        Dispositivo de destino.
    
    Returns
    -------
    torch.Tensor
        Tensor con los pesos de clase.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    labels_train: List[str] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    use_class_weights: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Entrena una CNN para clasificación binaria.
    
    Parameters
    ----------
    model : nn.Module
        Modelo de PyTorch.
    train_loader : DataLoader
        DataLoader de entrenamiento.
    val_loader : DataLoader
        DataLoader de validación.
    labels_train : List[str], optional
        Etiquetas de entrenamiento para calcular pesos de clase.
    epochs : int
        Número de épocas.
    lr : float
        Tasa de aprendizaje.
    use_class_weights : bool
        Si True, usa pesos de clase para manejar desbalance.
    verbose : bool
        Si True, imprime progreso.
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con historial de entrenamiento y modelo.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Configurar loss con pesos de clase si se solicita
    if use_class_weights and labels_train is not None:
        class_weights = get_class_weights(labels_train, device)
        pos_weight = class_weights[1] / class_weights[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if verbose:
            print(f"Usando pesos de clase: {class_weights}")
    else:
        criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            if use_class_weights and labels_train is not None:
                # Usar forward sin sigmoid (logits)
                if hasattr(model, 'forward_logits'):
                    preds = model.forward_logits(x)
                else:
                    preds = model(x)
            else:
                preds = model(x)
            
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validación
        val_metrics = evaluate_cnn(model, val_loader, device, verbose=False)
        history['val_loss'].append(val_metrics.get('loss', 0))
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {avg_train_loss:.4f} - "
                  f"Val Acc: {val_metrics['accuracy']:.4f} - "
                  f"Val F1: {val_metrics['f1']:.4f}")
    
    return {
        'model': model,
        'history': history,
        'device': device
    }


def evaluate_cnn(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evalúa un modelo CNN en un conjunto de datos.
    
    Parameters
    ----------
    model : nn.Module
        Modelo de PyTorch.
    data_loader : DataLoader
        DataLoader con los datos a evaluar.
    device : torch.device, optional
        Dispositivo de cómputo.
    verbose : bool
        Si True, imprime resultados.
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con métricas de evaluación.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, dtype=torch.float32)
            labels_cpu = labels.cpu().numpy().astype(int)
            
            outputs = model(images)
            probs = outputs.detach().cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            y_true.extend(labels_cpu.tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Métricas
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC y AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    if verbose:
        print("\n=== MÉTRICAS ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_true, y_pred, 
                                    target_names=["NORMAL", "PNEUMONIA"]))
    
    return {
        'accuracy': acc,
        'f1': f1,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'fpr': fpr,
        'tpr': tpr
    }


def save_model(
    model: nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = None,
    history: Dict = None
) -> None:
    """
    Guarda un modelo de PyTorch.
    
    Parameters
    ----------
    model : nn.Module
        Modelo a guardar.
    path : str
        Ruta de destino.
    optimizer : Optimizer, optional
        Optimizador a guardar.
    epoch : int, optional
        Época actual.
    history : Dict, optional
        Historial de entrenamiento.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if history is not None:
        checkpoint['history'] = history
    
    torch.save(checkpoint, path)
    print(f"Modelo guardado en: {path}")


def load_model(
    model: nn.Module,
    path: str,
    device: torch.device = None
) -> Tuple[nn.Module, Dict]:
    """
    Carga un modelo de PyTorch desde un checkpoint.
    
    Parameters
    ----------
    model : nn.Module
        Instancia del modelo (arquitectura).
    path : str
        Ruta del checkpoint.
    device : torch.device, optional
        Dispositivo de destino.
    
    Returns
    -------
    Tuple[nn.Module, Dict]
        (modelo cargado, checkpoint completo)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Modelo cargado desde: {path}")
    
    return model, checkpoint

