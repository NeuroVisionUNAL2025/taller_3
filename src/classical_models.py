# src/classical_models.py
"""
Modelos clásicos de Machine Learning para clasificación.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def get_default_models() -> Dict[str, Any]:
    """
    Retorna un diccionario con los modelos clásicos por defecto.
    
    Returns
    -------
    Dict[str, Any]
        Diccionario {nombre: modelo}.
    """
    return {
        "SVM-linear": SVC(kernel="linear", probability=True),
        "SVM-RBF": SVC(kernel="rbf", probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=300),
        "kNN-5": KNeighborsClassifier(n_neighbors=5),
        "LogReg": LogisticRegression(max_iter=1000)
    }


def evaluate_model(
    name: str,
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Entrena y evalúa un modelo de clasificación.
    
    Parameters
    ----------
    name : str
        Nombre del modelo.
    clf : estimator
        Clasificador de scikit-learn.
    X_train, y_train : np.ndarray
        Datos de entrenamiento.
    X_val, y_val : np.ndarray
        Datos de validación.
    X_test, y_test : np.ndarray
        Datos de prueba.
    verbose : bool
        Si True, imprime resultados.
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con métricas y modelo entrenado.
    """
    if verbose:
        print("-------------------------------------")
        print(f"\nModelo: {name}")

    clf.fit(X_train, y_train)

    # Validación
    pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, pred_val)
    f1_val = f1_score(y_val, pred_val)
    
    if verbose:
        print(f"Val Accuracy={acc_val:.4f}  F1={f1_val:.4f}")

    # Test
    pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, pred_test)
    f1_test = f1_score(y_test, pred_test)
    
    if verbose:
        print(f"TEST Accuracy={acc_test:.4f}  F1={f1_test:.4f}")
        print("Classification report TEST:")
        print(classification_report(y_test, pred_test))
        print("Confusion matrix TEST:")
        print(confusion_matrix(y_test, pred_test))

    return {
        "model": clf,
        "name": name,
        "val_accuracy": acc_val,
        "val_f1": f1_val,
        "test_accuracy": acc_test,
        "test_f1": f1_test,
        "predictions_test": pred_test,
        "confusion_matrix": confusion_matrix(y_test, pred_test)
    }


def train_all_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Entrena y evalúa múltiples modelos.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Diccionario {nombre: modelo}.
    X_train, y_train : np.ndarray
        Datos de entrenamiento.
    X_val, y_val : np.ndarray
        Datos de validación.
    X_test, y_test : np.ndarray
        Datos de prueba.
    verbose : bool
        Si True, imprime resultados.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Diccionario con resultados por modelo.
    """
    results = {}
    
    for name, clf in models.items():
        results[name] = evaluate_model(
            name, clf,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            verbose=verbose
        )
    
    return results


def cross_validate_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "f1",
    verbose: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    Realiza validación cruzada para múltiples modelos.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Diccionario {nombre: modelo}.
    X : np.ndarray
        Matriz de características.
    y : np.ndarray
        Vector de etiquetas.
    cv : int
        Número de folds.
    scoring : str
        Métrica de evaluación.
    verbose : bool
        Si True, imprime resultados.
    
    Returns
    -------
    Dict[str, Tuple[float, float]]
        Diccionario {nombre: (media, desviación)}.
    """
    results = {}
    
    for name, clf in models.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
        results[name] = (scores.mean(), scores.std())
        
        if verbose:
            print(f"\n{name} | CV {scoring} promedio: {scores.mean():.4f}  (+/- {scores.std():.4f})")
    
    return results


def get_feature_importance_rf(
    model: RandomForestClassifier,
    top_n: int = 20,
    verbose: bool = True
) -> np.ndarray:
    """
    Obtiene las características más importantes de un Random Forest.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Modelo entrenado.
    top_n : int
        Número de características top a mostrar.
    verbose : bool
        Si True, imprime resultados.
    
    Returns
    -------
    np.ndarray
        Índices de las características más importantes.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    if verbose:
        print(f"\nTop {top_n} features más importantes (Random Forest):")
        for i in indices[::-1]:
            print(f"Feature {i}: {importances[i]:.4f}")
    
    return indices[::-1]


def get_feature_importance_linear(
    model,
    top_n: int = 20,
    verbose: bool = True
) -> np.ndarray:
    """
    Obtiene las características más importantes de un modelo lineal (SVM o LogReg).
    
    Parameters
    ----------
    model : SVC or LogisticRegression
        Modelo entrenado con kernel lineal.
    top_n : int
        Número de características top a mostrar.
    verbose : bool
        Si True, imprime resultados.
    
    Returns
    -------
    np.ndarray
        Índices de las características más importantes.
    """
    coefs = np.abs(model.coef_)[0]
    indices = np.argsort(coefs)[-top_n:]
    
    if verbose:
        model_name = type(model).__name__
        print(f"\nTop {top_n} features más importantes ({model_name}):")
        for i in indices[::-1]:
            print(f"Feature {i}: {coefs[i]:.4f}")
    
    return indices[::-1]

