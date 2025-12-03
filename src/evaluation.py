# src/evaluation.py
"""
Funciones de evaluación y métricas para modelos de clasificación.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    auc
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    target_names: List[str] = None
) -> Dict[str, Any]:
    """
    Calcula métricas de clasificación completas.
    
    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Predicciones.
    y_prob : np.ndarray, optional
        Probabilidades de predicción (para ROC/AUC).
    target_names : List[str], optional
        Nombres de las clases.
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con todas las métricas.
    """
    if target_names is None:
        target_names = ["NORMAL", "PNEUMONIA"]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(
            y_true, y_pred, target_names=target_names
        )
    }
    
    # ROC y AUC si hay probabilidades
    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['thresholds'] = thresholds
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Imprime las métricas de clasificación de forma formateada.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Diccionario de métricas.
    """
    print("\n" + "="*50)
    print("MÉTRICAS DE CLASIFICACIÓN")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    if 'auc' in metrics:
        print(f"AUC:       {metrics['auc']:.4f}")
    
    print("\nMatriz de Confusión:")
    print(metrics['confusion_matrix'])
    
    print("\nReporte de Clasificación:")
    print(metrics['classification_report'])


def evaluate_sklearn_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evalúa un modelo de scikit-learn.
    
    Parameters
    ----------
    model : estimator
        Modelo de scikit-learn entrenado.
    X_test : np.ndarray
        Características de prueba.
    y_test : np.ndarray
        Etiquetas de prueba.
    verbose : bool
        Si True, imprime resultados.
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con métricas y predicciones.
    """
    y_pred = model.predict(X_test)
    
    # Obtener probabilidades si están disponibles
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None
    
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    metrics['y_pred'] = y_pred
    metrics['y_prob'] = y_prob
    
    if verbose:
        print_metrics(metrics)
    
    return metrics


def compare_models(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'f1'
) -> Tuple[str, Dict[str, float]]:
    """
    Compara múltiples modelos por una métrica específica.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Diccionario {nombre_modelo: resultados}.
    metric : str
        Métrica a comparar ('accuracy', 'f1', 'auc', etc.).
    
    Returns
    -------
    Tuple[str, Dict[str, float]]
        (nombre del mejor modelo, diccionario de scores)
    """
    scores = {}
    
    for name, res in results.items():
        if metric in res:
            scores[name] = res[metric]
        elif f'test_{metric}' in res:
            scores[name] = res[f'test_{metric}']
    
    if not scores:
        raise ValueError(f"Métrica '{metric}' no encontrada en los resultados")
    
    best_model = max(scores, key=scores.get)
    
    print(f"\nComparación de modelos por {metric}:")
    print("-" * 40)
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        marker = " *MEJOR*" if name == best_model else ""
        print(f"{name}: {score:.4f}{marker}")
    
    return best_model, scores


def get_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: List[str] = None
) -> Dict[str, List]:
    """
    Obtiene los índices y rutas de las muestras mal clasificadas.
    
    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Predicciones.
    paths : List[str], optional
        Lista de rutas a las imágenes.
    
    Returns
    -------
    Dict[str, List]
        Diccionario con índices y rutas de errores.
    """
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    result = {
        'indices': misclassified_idx.tolist(),
        'true_labels': y_true[misclassified_idx].tolist(),
        'pred_labels': y_pred[misclassified_idx].tolist(),
        'n_errors': len(misclassified_idx),
        'error_rate': len(misclassified_idx) / len(y_true)
    }
    
    if paths is not None:
        result['paths'] = [paths[i] for i in misclassified_idx]
    
    return result

