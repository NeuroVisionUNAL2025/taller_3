# src/visualization.py
"""
Utilidades de visualización para análisis y resultados.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any


def show_examples(
    paths: List[str],
    labels: List[str],
    preprocess_fn=None,
    n: int = 3,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Muestra ejemplos de imágenes originales y preprocesadas.
    
    Parameters
    ----------
    paths : List[str]
        Lista de rutas a las imágenes.
    labels : List[str]
        Lista de etiquetas.
    preprocess_fn : callable, optional
        Función de preprocesamiento.
    n : int
        Número de ejemplos a mostrar.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    plt.figure(figsize=figsize)
    idxs = np.random.choice(range(len(paths)), n, replace=False)

    for i, idx in enumerate(idxs):
        img = cv2.imread(paths[idx], cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, n, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{labels[idx]} (Original)")
        plt.axis('off')

        if preprocess_fn is not None:
            processed = preprocess_fn(paths[idx])
            plt.subplot(2, n, i + 1 + n)
            plt.imshow(processed, cmap='gray')
            plt.title(f"{labels[idx]} (Preprocesada)")
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    labels: List[Tuple[str, str]],
    figsize: Tuple[int, int] = (8, 5)
) -> None:
    """
    Grafica la distribución de clases en el dataset.
    
    Parameters
    ----------
    labels : List[Tuple[str, str]]
        Lista de tuplas (split, clase).
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    import pandas as pd
    
    df = pd.DataFrame({'label': labels})
    counts = df['label'].value_counts()
    
    plt.figure(figsize=figsize)
    counts.plot(kind='bar')
    plt.title('Distribución de clases')
    plt.xlabel('Clase (split, label)')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_image_sizes_distribution(
    sizes: List[Tuple[int, int]],
    figsize: Tuple[int, int] = (10, 4)
) -> None:
    """
    Grafica la distribución de tamaños de imágenes.
    
    Parameters
    ----------
    sizes : List[Tuple[int, int]]
        Lista de tuplas (ancho, alto).
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    ws = [s[0] for s in sizes]
    hs = [s[1] for s in sizes]

    plt.figure(figsize=figsize)
    plt.hist(ws, bins=30, alpha=0.7, label='Ancho')
    plt.hist(hs, bins=30, alpha=0.7, label='Alto')
    plt.title('Distribución de tamaños originales')
    plt.xlabel('Píxeles')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()
    
    print(f"Promedio tamaño: ({np.mean(ws):.1f}, {np.mean(hs):.1f})")
    print(f"Mediana tamaño: ({np.median(ws):.1f}, {np.median(hs):.1f})")


def plot_histograms_comparison(
    img_original: np.ndarray,
    img_processed: np.ndarray,
    titles: Tuple[str, str] = ('Original', 'Procesada'),
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Compara histogramas de dos imágenes.
    
    Parameters
    ----------
    img_original : np.ndarray
        Imagen original.
    img_processed : np.ndarray
        Imagen procesada.
    titles : Tuple[str, str]
        Títulos para cada histograma.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    if img_original.max() <= 1:
        img_original = (img_original * 255).astype('uint8')
    plt.hist(img_original.flatten(), bins=256)
    plt.title(f'Histograma {titles[0]}')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 2, 2)
    if img_processed.max() <= 1:
        img_processed = (img_processed * 255).astype('uint8')
    plt.hist(img_processed.flatten(), bins=256)
    plt.title(f'Histograma {titles[1]}')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.show()


def plot_hog_variations(
    img: np.ndarray,
    hog_fn,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Muestra variaciones del descriptor HOG con diferentes parámetros.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    hog_fn : callable
        Función para calcular HOG.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    bins_list = [12, 9, 6]
    cells_list = [(8, 8), (16, 16), (32, 32)]

    fig, axs = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle("Variaciones del Descriptor HOG", fontsize=16)

    # Variar bins
    for i, b in enumerate(bins_list):
        _, vis = hog_fn(img, pixels_per_cell=(16, 16), orientations=b)
        axs[0, i].imshow(vis, cmap='inferno')
        axs[0, i].set_title(f"cell=(16,16), bins={b}")
        axs[0, i].axis("off")

    # Variar tamaño de celda
    for i, c in enumerate(cells_list):
        _, vis = hog_fn(img, pixels_per_cell=c, orientations=9)
        axs[1, i].imshow(vis, cmap='inferno')
        axs[1, i].set_title(f"cell={c}, bins=9")
        axs[1, i].axis("off")

    # Combinaciones
    combos = list(zip(cells_list, bins_list))
    for i, (c, b) in enumerate(combos):
        _, vis = hog_fn(img, pixels_per_cell=c, orientations=b)
        axs[2, i].imshow(vis, cmap='inferno')
        axs[2, i].set_title(f"cell={c}, bins={b}")
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_lbp_images(
    img: np.ndarray,
    lbp_fn,
    configs: List[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Muestra imágenes LBP con diferentes configuraciones.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    lbp_fn : callable
        Función para calcular LBP.
    configs : List[Tuple[int, int]], optional
        Lista de configuraciones (P, R).
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    if configs is None:
        configs = [(8, 1), (16, 1), (24, 1),
                   (8, 1), (8, 2), (8, 3),
                   (8, 1), (16, 2), (24, 3)]

    n_configs = len(configs)
    ncols = 3
    nrows = (n_configs + ncols - 1) // ncols

    plt.figure(figsize=figsize)

    for i, (P, R) in enumerate(configs):
        lbp, hist = lbp_fn(img, P, R)
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(lbp, cmap='inferno')
        plt.title(f"LBP Image\nP={P}, R={R}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_lbp_histograms(
    img: np.ndarray,
    lbp_fn,
    configs: List[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Muestra histogramas LBP con diferentes configuraciones.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    lbp_fn : callable
        Función para calcular LBP.
    configs : List[Tuple[int, int]], optional
        Lista de configuraciones (P, R).
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    if configs is None:
        configs = [(8, 1), (16, 1), (24, 1),
                   (8, 1), (8, 2), (8, 3),
                   (8, 1), (16, 2), (24, 3)]

    n_configs = len(configs)
    ncols = 3
    nrows = (n_configs + ncols - 1) // ncols

    plt.figure(figsize=figsize)

    for i, (P, R) in enumerate(configs):
        lbp, hist = lbp_fn(img, P, R)
        plt.subplot(nrows, ncols, i + 1)
        plt.bar(np.arange(len(hist)), hist)
        plt.title(f"Histograma\nP={P}, R={R}")
        plt.xlabel("Patrones LBP")
        plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()


def plot_contour(
    img: np.ndarray,
    segmentar_fn,
    figsize: Tuple[int, int] = (5, 5)
) -> None:
    """
    Muestra el contorno detectado en una imagen.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises normalizada [0, 1].
    segmentar_fn : callable
        Función de segmentación.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    mask = segmentar_fn(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(cnts) == 0:
        print("No se encontraron contornos")
        return
    
    cnt = max(cnts, key=cv2.contourArea)

    img_draw = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_draw, [cnt], -1, (255, 0, 0), 2)

    plt.figure(figsize=figsize)
    plt.title("Contorno Detectado")
    plt.imshow(img_draw)
    plt.axis("off")
    plt.show()


def plot_fourier_descriptors(
    fd: np.ndarray,
    figsize: Tuple[int, int] = (7, 4)
) -> None:
    """
    Grafica las magnitudes de los descriptores de Fourier.
    
    Parameters
    ----------
    fd : np.ndarray
        Vector de descriptores de Fourier.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    plt.figure(figsize=figsize)
    plt.plot(np.abs(fd))
    plt.title("Fourier Descriptors — Magnitudes")
    plt.xlabel("Coeficiente")
    plt.ylabel("Magnitud")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (5, 4),
    title: str = "Matriz de Confusión"
) -> None:
    """
    Grafica una matriz de confusión.
    
    Parameters
    ----------
    cm : np.ndarray
        Matriz de confusión.
    class_names : List[str], optional
        Nombres de las clases.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    title : str
        Título de la figura.
    """
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, cmap="Blues", interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    
    n_classes = len(class_names)
    plt.xticks(range(n_classes), class_names)
    plt.yticks(range(n_classes), class_names)
    
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
    
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float = None,
    figsize: Tuple[int, int] = (6, 5),
    title: str = "ROC Curve"
) -> None:
    """
    Grafica la curva ROC.
    
    Parameters
    ----------
    fpr : np.ndarray
        Tasa de falsos positivos.
    tpr : np.ndarray
        Tasa de verdaderos positivos.
    roc_auc : float, optional
        Área bajo la curva.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    title : str
        Título de la figura.
    """
    if roc_auc is None:
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Grafica el historial de entrenamiento de una red neuronal.
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Diccionario con listas de métricas por época.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel('Época')
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def show_predictions(
    paths: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    preprocess_fn=None,
    n: int = 6,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Muestra ejemplos de predicciones con sus resultados.
    
    Parameters
    ----------
    paths : List[str]
        Lista de rutas a las imágenes.
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Predicciones.
    y_prob : np.ndarray
        Probabilidades de predicción.
    preprocess_fn : callable, optional
        Función de preprocesamiento.
    n : int
        Número de ejemplos a mostrar.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    """
    import random
    
    label_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
    indices = random.sample(range(len(paths)), min(n, len(paths)))
    
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel() if n > 1 else [axes]
    
    for ax, idx in zip(axes, indices):
        if preprocess_fn is not None:
            img = preprocess_fn(paths[idx])
        else:
            img = cv2.imread(paths[idx], cv2.IMREAD_GRAYSCALE)
        
        ax.imshow(img, cmap='gray')
        
        true_label = label_names[y_true[idx]]
        pred_label = label_names[y_pred[idx]]
        prob = y_prob[idx]
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        ax.set_title(f"Real: {true_label}\nPred: {pred_label} ({prob:.2f})", color=color)
        ax.axis('off')
    
    # Ocultar ejes vacíos
    for ax in axes[len(indices):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

