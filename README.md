# Taller 3 — Clasificación de Imágenes Médicas

Proyecto académico de clasificación de imágenes de rayos X de tórax para detección de neumonía. Implementa y compara un pipeline de **técnicas clásicas de extracción de características** (HOG, Hu moments, LBP, GLCM, Gabor, Fourier) con clasificadores de Machine Learning (SVM, Random Forest, kNN, Regresión Logística) contra un enfoque de **Deep Learning** con redes neuronales convolucionales (CNN).

Documentación ampliada en `docs/index.md`.

### Objetivo
- Comparar el rendimiento de técnicas clásicas de visión por computador vs Deep Learning en un problema de clasificación binaria de imágenes médicas.
- Implementar un pipeline reproducible: carga de datos → preprocesamiento → extracción de características → clasificación → evaluación.
- Analizar métricas de rendimiento (Accuracy, F1, AUC-ROC) y la importancia de las características.

## Estructura del repositorio

```text
taller_3/
├─ data/
│  └─ chest_xray/             # Dataset de rayos X (NORMAL vs PNEUMONIA)
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ docs/                       # Informe (GitHub Pages/Jekyll)
│  └─ index.md                 # Informe principal con resultados y figuras
├─ models/                     # Modelos entrenados guardados
├─ notebooks/                  # Cuadernos de trabajo
│  ├─ 01_exploratory_analysis.ipynb
│  ├─ 02_feature_extraction.ipynb
│  └─ 03_classification_pipeline.ipynb
├─ results/                    # Salidas (figuras, métricas)
├─ src/                        # Módulos reutilizables del proyecto
│  ├─ __init__.py              # Inicialización del paquete
│  ├─ data_loader.py           # load_image_paths, split_by_set, labels_to_numeric
│  ├─ preprocessing.py         # read_and_preprocess, apply_clahe, normalize_image
│  ├─ descriptors/             # Descriptores de características
│  │  ├─ __init__.py
│  │  ├─ shape.py              # descriptor_hog, descriptor_hu, descriptor_contorno, descriptor_fourier
│  │  ├─ texture.py            # descriptor_lbp, descriptor_glcm, descriptor_gabor
│  │  └─ statistics.py         # first_order_stats
│  ├─ features.py              # extract_features_all, build_feature_matrix, normalize_features, apply_pca
│  ├─ classical_models.py      # evaluate_model, train_all_models, cross_validate_models
│  ├─ deep_learning/           # Módulos de Deep Learning
│  │  ├─ __init__.py
│  │  ├─ models.py             # SimpleCNN
│  │  ├─ dataset.py            # ChestXrayDataset, create_dataloaders
│  │  └─ training.py           # train_cnn, evaluate_cnn, save_model, load_model
│  ├─ evaluation.py            # compute_metrics, compare_models, get_misclassified_samples
│  ├─ visualization.py         # plot_confusion_matrix, plot_roc_curve, show_predictions, etc.
│  └─ base/                    # Scripts base de desarrollo
├─ requirements.txt            # Dependencias de ejecución
└─ README.md
```

## Dataset

El proyecto utiliza el dataset **Chest X-Ray Images (Pneumonia)** de Kaggle:
- **Clases**: NORMAL, PNEUMONIA
- **Estructura**: train/val/test con imágenes en formato JPEG
- **Preprocesamiento**: Redimensionado a 256×256, CLAHE para mejora de contraste

## Requisitos
- Python 3.10 o superior
- Windows 10/11 (probado). También funciona en Linux/macOS
- GPU con CUDA (opcional, para acelerar Deep Learning)

## Instalación y entorno (Windows PowerShell)

```powershell
# 1) Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate

# 2) Actualizar pip
python -m pip install --upgrade pip

# 3) Instalar dependencias
pip install -r requirements.txt
```

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Módulos principales

### Carga de datos (`data_loader.py`)
```python
from src.data_loader import load_image_paths, split_by_set, labels_to_numeric

paths, labels = load_image_paths("data/chest_xray")
(paths_train, labels_train), (paths_val, labels_val), (paths_test, labels_test) = split_by_set(paths, labels)
y_train = labels_to_numeric(labels_train)
```

### Preprocesamiento (`preprocessing.py`)
```python
from src.preprocessing import read_and_preprocess

img = read_and_preprocess("path/to/image.jpeg", img_size=(256, 256), apply_clahe=True)
```

### Descriptores de características (`descriptors/`)
```python
from src.descriptors import (
    descriptor_hog,      # Histogram of Oriented Gradients
    descriptor_hu,       # 7 momentos de Hu (invariantes)
    descriptor_lbp,      # Local Binary Patterns
    descriptor_glcm,     # Gray Level Co-occurrence Matrix
    descriptor_gabor,    # Filtros de Gabor
    descriptor_fourier   # Descriptores de Fourier del contorno
)

hog_vec, hog_img = descriptor_hog(img)
hu_moments = descriptor_hu(img)
lbp_img, lbp_hist = descriptor_lbp(img, P=8, R=2)
```

### Extracción de características (`features.py`)
```python
from src.features import build_feature_matrix, normalize_features, apply_pca

X_train = build_feature_matrix(paths_train)
X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(X_train, X_val, X_test)
X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(X_train_norm, X_val_norm, X_test_norm)
```

### Modelos clásicos (`classical_models.py`)
```python
from src.classical_models import get_default_models, train_all_models

models = get_default_models()  # SVM, RF, kNN, LogReg
results = train_all_models(models, X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test)
```

### Deep Learning (`deep_learning/`)
```python
from src.deep_learning import SimpleCNN, ChestXrayDataset, create_dataloaders, train_cnn, evaluate_cnn

# Crear dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    paths_train, labels_train,
    paths_val, labels_val,
    paths_test, labels_test,
    batch_size=32
)

# Entrenar CNN
model = SimpleCNN()
result = train_cnn(model, train_loader, val_loader, labels_train, epochs=50)

# Evaluar
metrics = evaluate_cnn(model, test_loader)
```

### Visualización (`visualization.py`)
```python
from src.visualization import plot_confusion_matrix, plot_roc_curve, show_predictions

plot_confusion_matrix(metrics['confusion_matrix'])
plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'])
```

## Cómo ejecutar

### Opción A: Cuadernos Jupyter
- Abre los notebooks en tu IDE (VS Code recomendado) o Jupyter Lab
- Cuadernos principales:
  - `notebooks/01_exploratory_analysis.ipynb`: análisis exploratorio
  - `notebooks/02_feature_extraction.ipynb`: extracción de características
  - `notebooks/03_classification_pipeline.ipynb`: pipeline completo de clasificación

### Opción B: Script de ejemplo
```python
from src.data_loader import load_image_paths, split_by_set, labels_to_numeric
from src.features import build_feature_matrix, normalize_features, apply_pca
from src.classical_models import get_default_models, train_all_models

# Cargar datos
paths, labels = load_image_paths("data/chest_xray")
(paths_train, labels_train), (paths_val, labels_val), (paths_test, labels_test) = split_by_set(paths, labels)

# Convertir etiquetas
y_train = labels_to_numeric(labels_train)
y_val = labels_to_numeric(labels_val)
y_test = labels_to_numeric(labels_test)

# Extraer características
X_train = build_feature_matrix(paths_train)
X_val = build_feature_matrix(paths_val)
X_test = build_feature_matrix(paths_test)

# Normalizar y reducir dimensionalidad
X_train_norm, X_val_norm, X_test_norm, _ = normalize_features(X_train, X_val, X_test)
X_train_pca, X_val_pca, X_test_pca, _ = apply_pca(X_train_norm, X_val_norm, X_test_norm)

# Entrenar y evaluar modelos
models = get_default_models()
results = train_all_models(models, X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test)
```

## Técnicas implementadas

### Descriptores clásicos
| Descriptor | Tipo | Características |
|------------|------|-----------------|
| HOG | Forma/Gradiente | Histograma de gradientes orientados |
| Hu Moments | Forma | 7 momentos invariantes a transformaciones |
| LBP | Textura | Patrones binarios locales |
| GLCM | Textura | Matriz de co-ocurrencia de niveles de gris |
| Gabor | Textura | Filtros de Gabor a múltiples frecuencias |
| Fourier | Forma | Descriptores del contorno en frecuencia |

### Clasificadores
| Modelo | Tipo |
|--------|------|
| SVM (Linear) | Máquinas de vectores de soporte |
| SVM (RBF) | SVM con kernel gaussiano |
| Random Forest | Ensemble de árboles |
| kNN | k vecinos más cercanos |
| Logistic Regression | Regresión logística |
| SimpleCNN | Red neuronal convolucional |

## Resultados esperados
- Comparación de métricas (Accuracy, F1, AUC) entre enfoques clásicos y Deep Learning
- Análisis de importancia de características
- Matrices de confusión y curvas ROC
- Visualización de predicciones correctas e incorrectas

## Créditos
Equipo UNAL — Curso de Visión por Computador. Informe completo y resultados en `docs/index.md`.
