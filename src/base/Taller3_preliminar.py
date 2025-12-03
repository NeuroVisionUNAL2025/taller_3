# IMPORTS
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage.feature import hog
from skimage.feature import hog
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Cargar el dataset y ruta
DATA_DIR = r"./chest_xray/chest_xray"

def load_image_paths(data_dir):
    classes = ['NORMAL', 'PNEUMONIA']
    paths = []
    labels = []

    for split in ['train', 'val', 'test']:
        for cls in classes:
            folder = os.path.join(data_dir, split, cls)
            if not os.path.exists(folder):
                continue

            for p in glob(os.path.join(folder, '*.jpeg')) + \
                     glob(os.path.join(folder, '*.jpg')) + \
                     glob(os.path.join(folder, '*.png')):
                paths.append(p)
                labels.append((split, cls))
    return paths, labels

paths, labels = load_image_paths(DATA_DIR)

#Distribución de clases
df = pd.DataFrame({'path': paths, 'label': labels})
print(df['label'].value_counts())

df['label'].value_counts().plot(kind='bar', figsize=(5,4))
plt.title('Distribución de clases')
plt.show()

#Análisis de tamaño de imagenes
sizes = []
for p in paths:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    sizes.append((w, h))

ws = [s[0] for s in sizes]
hs = [s[1] for s in sizes]

print("Promedio tamaño:", (np.mean(ws), np.mean(hs)))
print("Mediana tamaño:", (np.median(ws), np.median(hs)))

plt.figure(figsize=(10,4))
plt.hist(ws, bins=30, alpha=0.7, label='Ancho')
plt.hist(hs, bins=30, alpha=0.7, label='Alto')
plt.title('Distribución de tamaños originales')
plt.legend()
plt.show()

#Reescalado y CLAHE
IMG_SIZE = (256, 256)

def read_and_preprocess(path, img_size=IMG_SIZE, apply_clahe=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer: {path}")

    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    return img

def show_examples(paths, labels, n=3):
    plt.figure(figsize=(12, 6))
    idxs = np.random.choice(range(len(paths)), n, replace=False)

    for i, idx in enumerate(idxs):
        img = cv2.imread(paths[idx], cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, n, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{labels[idx]} (Original)")
        plt.axis('off')

        processed = read_and_preprocess(paths[idx], apply_clahe=True)
        plt.subplot(2, n, i+1+n)
        plt.imshow(processed, cmap='gray')
        plt.title(f"{labels[idx]} (CLAHE)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

show_examples(paths, labels, n=3)

#Histogramas comparativos antes y después de CLAHE
sample_path = paths[0]

img_original = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
img_clahe = read_and_preprocess(sample_path, apply_clahe=True)
img_clahe_uint8 = (img_clahe * 255).astype('uint8')

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.hist(img_original.flatten(), bins=256)
plt.title('Histograma Original')

plt.subplot(1,2,2)
plt.hist(img_clahe_uint8.flatten(), bins=256)
plt.title('Histograma con CLAHE')

plt.show()

#Carga de imagen
img = read_and_preprocess(sample_path)

#Función imagen muestra
def ver_img_muestra(img):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Imagen preprocesada")
    plt.imshow(img, cmap='gray')
    plt.axis("off")

def descriptor_hog(img, pixels_per_cell=(16,16), orientations=9):
    hog_vec, hog_img = hog(
        (img*255).astype("uint8"),
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2,2),
        visualize=True
    )
    return hog_vec, hog_img

def grid_hog_variations(img):
    
    bins_list = [12, 9, 6]
    cells_list = [(8,8), (16,16), (32,32)]

    fig, axs = plt.subplots(3, 3, figsize=(12,12))
    fig.suptitle("Variaciones del Descriptor HOG", fontsize=16)

    # variar bins
    for i, b in enumerate(bins_list):
        _, vis = descriptor_hog(img, pixels_per_cell=(16,16), orientations=b)
        axs[0, i].imshow(vis, cmap='inferno')
        axs[0, i].set_title(f"cell={16,16}, bins={b}")
        axs[0, i].axis("off")

    # variar tamaño de celda
    for i, c in enumerate(cells_list):
        _, vis = descriptor_hog(img, pixels_per_cell=c, orientations=9)
        axs[1, i].imshow(vis, cmap='inferno')
        axs[1, i].set_title(f"cell={c}, bins={9}")
        axs[1, i].axis("off")

    # combinaciones

    combos = list(zip(cells_list, bins_list)) 
    for i, (c, b) in enumerate(combos):
        _, vis = descriptor_hog(img, pixels_per_cell=c, orientations=b)
        axs[2, i].imshow(vis, cmap='inferno')
        axs[2, i].set_title(f"cell={c}, bins={b}")
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.show()

ver_img_muestra(img)
grid_hog_variations(img)

def descriptor_hu(img):
    img_u8 = (img*255).astype("uint8")
    m = cv2.moments(img_u8)
    hu = cv2.HuMoments(m).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log

def test_invariances(img):
    # Traslación
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    shift = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    print("\nTraslación:", descriptor_hu(shift))

    # Rotación
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 90, 1)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    print("\nRotación 90°:", descriptor_hu(rot))

    # Escalado
    scale = cv2.resize(img, None, fx=0.5, fy=0.5)
    print("\nEscalado 0.5x:", descriptor_hu(scale))

print("\n7 momentos HU: ", descriptor_hu(img))
print(test_invariances(img))

def segmentar(img):
    img_u8 = (img*255).astype("uint8")
    _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def descriptor_contorno(img):
    mask = segmentar(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return {"area":0, "perimetro":0, "circularidad":0, "excentricidad":0}

    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    circularidad = 4 * np.pi * area / (perimetro**2 + 1e-12)

    # Excentricidad vía elipse ajustada
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        excentricidad = np.sqrt(1 - (MA/ma)**2)
    else:
        excentricidad = np.nan

    return {
        "area": area,
        "perimetro": perimetro,
        "circularidad": circularidad,
        "excentricidad": excentricidad
    }

def show_contour(img):
    mask = segmentar(img)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    img_draw = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_draw, [cnt], -1, (255,0,0), 2)

    plt.figure(figsize=(5,5))
    plt.title("Contorno Detectado")
    plt.imshow(img_draw)
    plt.axis("off")
    plt.show()

show_contour(img)
descriptor_contorno(img)

def descriptor_fourier(img, n_coeff=20):
    mask = segmentar(img)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(cnts) == 0:
        return np.zeros(n_coeff)

    cnt = max(cnts, key=len).squeeze()
    contorno = cnt[:,0] + 1j*cnt[:,1]

    fd = np.fft.fft(contorno)

    # Normalización (invarianza a escala)
    fd_norm = fd / (np.abs(fd[1]) + 1e-9)

    return fd_norm[:n_coeff]

def plot_fourier(fd):
    plt.figure(figsize=(7,4))
    plt.plot(np.abs(fd))
    plt.title("Fourier Descriptors — Magnitudes")
    plt.grid(True)
    plt.show()

fd = descriptor_fourier(img, n_coeff=10)
plot_fourier(fd)
print("FD Shape:", fd.shape)

from skimage.feature import local_binary_pattern

def descriptor_lbp(img, P=8, R=1):
    lbp = local_binary_pattern(img, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(P + 3), density=True)
    return lbp, hist

def show_lbp_img(img):
    configs = [(8,1), (16,1), (24,1),
               (8,1), (8,2), (8,3),
               (8,1), (16,2), (24,3)]

    plt.figure(figsize=(12,12))
    
    for i, (P,R) in enumerate(configs):
        lbp, hist = descriptor_lbp(img, P, R)
        # Imagen LBP
        plt.subplot(3,3,i+1)
        plt.imshow(lbp, cmap='inferno')
        plt.title(f"LBP Image\nP={P}, R={R}")
        plt.axis("off")

    plt.tight_layout()   
    plt.show()

def show_lbp_hist(img):
    configs = [(8,1), (16,1), (24,1),
               (8,1), (8,2), (8,3),
               (8,1), (16,2), (24,3)]

    plt.figure(figsize=(12,12))

    for i, (P,R) in enumerate(configs):
        lbp, hist = descriptor_lbp(img, P, R)
        # Histograma
        plt.subplot(3,3,i+1)
        plt.bar(np.arange(len(hist)), hist)
        plt.title(f"Histograma\nP={P}, R={R}")
        plt.xlabel("Patrones LBP")
        plt.ylabel("Frecuencia")

    plt.tight_layout()   
    plt.show()

ver_img_muestra(img)
show_lbp_img(img)
show_lbp_hist(img)

from skimage.feature import graycomatrix, graycoprops

def descriptor_glcm(img, distances=[1], angles=[0], levels=256,
                    props=['contrast', 'correlation', 'energy', 'homogeneity']):
    img_u8 = (img * 255).astype('uint8')

    glcm = graycomatrix(
        img_u8,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    feats = []
    for p in props:
        vals = graycoprops(glcm, p)
        feats.append(vals.mean())

    return np.array(feats)

descriptor_glcm(img)

from skimage.filters import gabor

def descriptor_gabor(img, freqs=[0.1, 0.2, 0.3], thetas=[0, np.pi/4, np.pi/2]):
    feats = []
    for f in freqs:
        for t in thetas:
            real, imag = gabor(img, frequency=f, theta=t)
            feats.append(real.mean())
            feats.append(real.std())
    return np.array(feats)

from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy

def first_order_stats(img):
    img_f = img.astype(np.float32).ravel()
    feats = {
        "mean": img_f.mean(),
        "var": img_f.var(),
        "skew": skew(img_f),
        "kurtosis": kurtosis(img_f),
        "entropy": shannon_entropy(img)
    }
    return feats

#Separación en train, val y test

paths_train = [p for p, (split, cls) in zip(paths, labels) if split == 'train']
labels_train = [cls for split, cls in labels if split == 'train']

paths_val = [p for p, (split, cls) in zip(paths, labels) if split == 'val']
labels_val = [cls for split, cls in labels if split == 'val']

paths_test = [p for p, (split, cls) in zip(paths, labels) if split == 'test']
labels_test = [cls for split, cls in labels if split == 'test']

# Etiquetas de texto a números (NORMAL=0, PNEUMONIA=1)
label_map = {'NORMAL': 0, 'PNEUMONIA': 1}

y_train = np.array([label_map[l] for l in labels_train])
y_val = np.array([label_map[l] for l in labels_val])
y_test = np.array([label_map[l] for l in labels_test])

print("Train:", len(paths_train), len(labels_train))
print("Val:", len(paths_val), len(labels_val))
print("Test:", len(paths_test), len(labels_test))

def extract_features_all(img, descriptor_hog, descriptor_hu, descriptor_fourier,
                         descriptor_lbp, descriptor_glcm, descriptor_gabor):

    img_u8 = (img * 255).astype("uint8")
    feats = []

    # HOG
    hog_vec, _ = descriptor_hog(img)
    feats.extend(hog_vec.flatten().tolist())

    # Hu
    hu = descriptor_hu(img)
    feats.extend(hu.tolist())

    # Fourier
    coeff = descriptor_fourier(img)
    feats.extend(np.abs(coeff).tolist())

    # LBP
    _, lbp_hist = descriptor_lbp(img, P=8, R=2)
    feats.extend(lbp_hist.flatten().tolist())

    # GLCM
    glcm_feats = descriptor_glcm(img_u8)
    feats.extend(glcm_feats.tolist())

    # Gabor
    gabor_feats = descriptor_gabor(img)
    feats.extend(np.array(gabor_feats, dtype=np.float32).flatten().tolist())

    return np.array(feats, dtype=np.float32)

def build_feature_matrix(paths, descriptor_hog, descriptor_hu, descriptor_fourier,
                         descriptor_lbp, descriptor_glcm, descriptor_gabor):

    X = []

    for p in tqdm(paths):
        img = read_and_preprocess(p)
        if img is None:
            print(f"Imagen inválida: {p}")
            continue

        try:
            features = extract_features_all(
                img,
                descriptor_hog, descriptor_hu, descriptor_fourier,
                descriptor_lbp, descriptor_glcm, descriptor_gabor
            )

            if features.ndim != 1:
                print(f"Vector con shape raro en {p}: {features.shape}")
                continue

            X.append(features)

        except Exception as e:
            print(f"Error al procesar {p}: {e}")

    return np.array(X, dtype=np.float32)

X_train = build_feature_matrix(
    paths_train,
    descriptor_hog, descriptor_hu, descriptor_fourier,
    descriptor_lbp, descriptor_glcm, descriptor_gabor
)

X_val = build_feature_matrix(
    paths_val,
    descriptor_hog, descriptor_hu, descriptor_fourier,
    descriptor_lbp, descriptor_glcm, descriptor_gabor
)

X_test = build_feature_matrix(
    paths_test,
    descriptor_hog, descriptor_hu, descriptor_fourier,
    descriptor_lbp, descriptor_glcm, descriptor_gabor
)

y_train = np.array([label_map[l] for l in labels_train], dtype=np.int32)
y_val   = np.array([label_map[l] for l in labels_val], dtype=np.int32)
y_test  = np.array([label_map[l] for l in labels_test], dtype=np.int32)

#Normalización
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

#Reducción de dimensionalidad
from sklearn.decomposition import PCA

# Reducir la dimensionalidad 
pca = PCA(0.95)

# PCA en el conjunto de entrenamiento normalizado
pca.fit(X_train_norm)

# Aplicar la transformación a los tres conjuntos
X_train_pca = pca.transform(X_train_norm)
X_val_pca = pca.transform(X_val_norm)
X_test_pca = pca.transform(X_test_norm)

# Entrenamiento y evaluación en val y test
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Evaluación de modelos
def evaluate_model(name, clf, X_train, y_train, X_val, y_val, X_test, y_test):
    print("-------------------------------------")
    print("\nModelo:", name)

    clf.fit(X_train, y_train)

    # Validation
    pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, pred_val)
    f1_val  = f1_score(y_val, pred_val)
    print(f"Val Accuracy={acc_val:.4f}  F1={f1_val:.4f}")

    # Test
    pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, pred_test)
    f1_test  = f1_score(y_test, pred_test)
    print(f"TEST Accuracy={acc_test:.4f}  F1={f1_test:.4f}")

    print("Classification report TEST:")
    print(classification_report(y_test, pred_test))

    print("Confusion matrix TEST:")
    print(confusion_matrix(y_test, pred_test))

    return clf

# Modelos a entrenar
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


models = {
    "SVM-linear": SVC(kernel="linear", probability=True),
    "SVM-RBF": SVC(kernel="rbf", probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=300),
    "kNN-5": KNeighborsClassifier(n_neighbors=5),
    "LogReg": LogisticRegression(max_iter=1000)
}

# entrenamiento del modelo
trained_models = {}

for name, clf in models.items():
    trained_models[name] = evaluate_model(name, clf,
                                          X_train_pca, y_train,
                                          X_val_pca, y_val,
                                          X_test_pca, y_test)

from sklearn.model_selection import cross_val_score

for name, clf in models.items():
    scores = cross_val_score(clf, X_train_pca, y_train, cv=5, scoring="f1")
    print(f"\n{name} | CV F1 promedio: {scores.mean():.4f}  (+/- {scores.std():.4f})")

# RandomForest
rf = trained_models["RandomForest"]

importances = rf.feature_importances_

# Mostrar top 20
indices = np.argsort(importances)[-20:]

print("\nTop 20 features más importantes (Random Forest):")
for i in indices[::-1]:
    print(f"Feature {i}: {importances[i]:.4f}")

# SVM 
svm_lin = trained_models["SVM-linear"]
coefs = np.abs(svm_lin.coef_)[0]

indices = np.argsort(coefs)[-20:]

print("\nTop 20 features más importantes (SVM Linear):")
for i in indices[::-1]:
    print(f"Feature {i}: {coefs[i]:.4f}")

print(trained_models.keys())

# Regresión logística
lr = trained_models["LogReg"]
coefs = np.abs(lr.coef_)[0]

indices = np.argsort(coefs)[-20:]

print("\nTop 20 features más importantes (Logistic Regression):")
for i in indices[::-1]:
    print(f"Feature {i}: {coefs[i]:.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x)
 


class ChestXrayDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        self.label_map = {"NORMAL": 0.0, "PNEUMONIA": 1.0}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = transform(img)  
        
        label_str = self.labels[idx]
        label = self.label_map[label_str]

        return img, torch.tensor(label, dtype=torch.float32)

def train_cnn(model, train_loader, val_loader, labels_train, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calcular pesos de clase
    classes = np.unique(labels_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels_train
    )
    
    # Convertir a tensor de PyTorch
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Pesos de clase:", class_weights_tensor)
    
    # Calcular pos_weight para BCEWithLogitsLoss
    # Asumimos: clase 0 = NORMAL, clase 1 = PNEUMONIA
    pos_weight = class_weights_tensor[1] / class_weights_tensor[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # Asegurar forma [batch, 1]

            optimizer.zero_grad()
            preds = model(x)  # Salida sin sigmoid (logits)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def evaluate_model_on_test(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        y_prob = clf.decision_function(X_test)

    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    print("AUC:", auc)

    return fpr, tpr, auc

# Crear datasets
from PIL import Image

train_dataset = ChestXrayDataset(paths_train, labels_train)
val_dataset = ChestXrayDataset(paths_val, labels_val)
test_dataset = ChestXrayDataset(paths_test, labels_test)

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SimpleCNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("\nEntrenando CNN...\n")
for epoch in range(50):
    model.train()
    loss_total = 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds,y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    print(f"Epoch {epoch+1}/5 — Loss: {loss_total/len(train_loader):.4f}")

# CELDA: Evaluación completa (pegar y ejecutar)
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, f1_score
import matplotlib.pyplot as plt
import random

# 1) Device (define GPU si está, si no CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# 2) Asegúrate de que el modelo esté en el device
model = model.to(device)
model.eval()

# 3) Recolectar predicciones en el conjunto de validación
y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device, dtype=torch.float32)
        # labels pueden venir como float tensor; asegurar dtype y forma
        labels_cpu = labels.cpu().numpy().astype(int)
        outputs = model(images)                  # salida sigmoid ya en forward
        probs = outputs.detach().cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        y_true.extend(labels_cpu.tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# 4) Métricas
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("\n=== METRICAS ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1: {f1:.4f}")
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=["NORMAL","PNEUMONIA"]))

# 5) Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues", interpolation='nearest')
plt.title("Matriz de Confusión (Val)")
plt.colorbar()
plt.xticks([0,1], ["NORMAL","PNEUMONIA"])
plt.yticks([0,1], ["NORMAL","PNEUMONIA"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="red")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

# 6) ROC y AUC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Val)")
plt.legend()
plt.show()
print(f"AUC: {roc_auc:.4f}")

# 7) Mostrar ejemplos aleatorios con predicción
n_show = 6
indices = random.sample(range(len(paths_val)), n_show)
fig, axs = plt.subplots(2, 3, figsize=(12,8))
for ax, idx in zip(axs.ravel(), indices):
    p = paths_val[idx]
    img = read_and_preprocess(p)   # devuelve grayscale [0,1]
    # buscar pred y prob correspondiente: necesitamos index en val_paths
    # construimos mapping si hace falta
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Real: {labels_val[idx]}  Pred: { 'PNEUMONIA' if y_pred[idx]==1 else 'NORMAL' } ({y_prob[idx]:.2f})")
    ax.axis('off')
plt.show()

# 8) Guardar modelo (opcional)
save_path = "modelo_simple_cnn.pt"
torch.save(model.state_dict(), save_path)
print("Modelo guardado en:", save_path)

