# Taller 3 — Visión por Computador (MODIFICAR)

Proyecto académico de registro y fusión de imágenes (mosaico panorámico) con medición métrica sobre el resultado. Implementa un pipeline con detección de características (SIFT, ORB, AKAZE), emparejamiento y filtrado (ratio test de Lowe), estimación de homografías con RANSAC, cosido con blending tipo feather y una interfaz sencilla para calibración y medición.

Documentación ampliada en `docs/index.md`.

### Objetivo
- Integrar un pipeline reproducible para: detección → descripción → matching → registro → fusión → calibración métrica → medición.
- Validar el pipeline con imágenes sintéticas y aplicar en un caso real (tres vistas de un comedor).

## Estructura del repositorio

```text
taller_2_cv/
├─ data/
│  └─ original/                # Imágenes de entrada (ejemplo: IMG01.jpg, IMG02.jpg, IMG03.jpg)
├─ docs/                       # Informe (GitHub Pages/Jekyll)
│  └─ index.md                 # Informe principal con resultados y figuras
├─ notebooks/                  # Cuadernos de trabajo
│  ├─ 01_exploratory_analysis.ipynb
│  ├─ 02_synthetic_validation.ipynb
│  └─ 03_main_pipeline.ipynb   # Pipeline principal (registro, mosaico y medición)
├─ results/                    # Salidas (figuras, mediciones)
├─ src/                        # Módulos reutilizables del proyecto
│  ├─ feature_detection.py     # detect_and_describe(...): SIFT/ORB/AKAZE
│  ├─ matching.py              # match_descriptors(...), keypoints_to_points(...)
│  ├─ registration.py          # estimate_homography(...), stitch_images_blend(...)
│  ├─ measurement.py           # set_scale_by_two_points(...), measure_distance(...), interactive_pick_points(...)
│  ├─ utils.py                 # utilidades de IO/visualización y generación sintética
│  └─ tests/                   # pruebas unitarias (pytest)
├─ requirements.txt            # dependencias de ejecución (incluye opencv-contrib)
├─ requirements-dev.txt        # dependencias de desarrollo (pytest, cobertura)
└─ README.md
```

## Requisitos
- Python 3.10 o superior.
- Windows 10/11 (probado). También funciona en Linux/macOS con ajustes menores.
- OpenCV con módulos contrib (incluido vía `opencv-contrib-python` en `requirements.txt`).

## Instalación y entorno (Windows PowerShell)

```powershell
# 1) Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate

# 2) Actualizar pip
python -m pip install --upgrade pip

# 3) Instalar dependencias de ejecución
pip install -r requirements.txt

# (Opcional) Dependencias de desarrollo (para ejecutar pruebas)
pip install -r requirements-dev.txt
```

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # opcional
```

## Cómo ejecutar

### Opción A: Cuadernos Jupyter
- Abre los notebooks en tu IDE (VS Code recomendado) o instala Jupyter Lab:

```bash
pip install jupyterlab  # solo si deseas ejecutarlos fuera del IDE
```

- Cuadernos principales:
  - `notebooks/02_synthetic_validation.ipynb`: validación con pares sintéticos.
  - `notebooks/03_main_pipeline.ipynb`: pipeline real sobre `data/original/` y medición.

Sugerencia: si usas Jupyter clásico, registra el kernel del entorno virtual:

```bash
python -m ipykernel install --user --name taller2-cv
```

### Opción B: Usar los módulos de `src/` desde un script
Ejemplo mínimo para coser tres imágenes y crear el mosaico:

```python
from pathlib import Path
import cv2 as cv
from src.utils import imread_color
from src.feature_detection import detect_and_describe
from src.matching import match_descriptors
from src.registration import estimate_homography, stitch_images_blend

root = Path(__file__).resolve().parent
imgs = [
    imread_color(root / 'data' / 'original' / 'IMG01.jpg'),
    imread_color(root / 'data' / 'original' / 'IMG02.jpg'),
    imread_color(root / 'data' / 'original' / 'IMG03.jpg'),
]

# Referencia: imagen 0
kps0, desc0 = detect_and_describe(imgs[0], method='SIFT')
Hs = []
for i in (1, 2):
    kpsi, desci = detect_and_describe(imgs[i], method='SIFT')
    good = match_descriptors(desci, desc0, strategy='BF', ratio=0.75)
    H, _ = estimate_homography(kpsi, kps0, good, ransac_thresh=3.0)
    Hs.append(H)

panorama = stitch_images_blend([imgs[0], imgs[1], imgs[2]], Hs, blend='feather')
out_path = root / 'results' / 'figures' / '03_main_pipeline' / 'mosaico_selected_SIFT.jpg'
out_path.parent.mkdir(parents=True, exist_ok=True)
cv.imwrite(str(out_path), panorama)
print(f'Mosaico guardado en: {out_path}')
```

Guarda el fragmento como `run_pipeline.py` en la raíz del repo y ejecútalo:

```powershell
python run_pipeline.py
```

### Medición métrica (interactivo)
Para fijar escala y medir distancias en el mosaico usa `measurement.py` (ver `notebooks/03_main_pipeline.ipynb`). La función `interactive_pick_points(...)` abre una ventana de OpenCV para seleccionar puntos.

## Pruebas

```bash
pytest -q
```

## Problemas comunes
- “SIFT no disponible”: asegúrate de haber instalado `opencv-contrib-python` (incluido en `requirements.txt`).
- Ventanas de OpenCV detrás de otras: en Windows se intenta llevar al frente; si no aparece, revisa ventanas minimizadas.
- En WSL, las ventanas de OpenCV pueden no mostrarse (usa notebooks o un entorno con servidor X).

## Créditos
Equipo UNAL — Curso de Visión por Computador. Informe completo y resultados en `docs/index.md`.