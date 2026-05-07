# 🧠 Clasificación y Segmentación de Tumores Cerebrales en MRI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

Trabajo Práctico Final — Visión por Computadora  
**Autores:** Mauro Virgilio Blanc · Juan Pablo Imbrigno · Andrea Ferenaz  
**Año:** 2025

---

## 📋 Descripción

Sistema de diagnóstico asistido por inteligencia artificial para la detección y segmentación de tumores cerebrales en imágenes de resonancia magnética (MRI). El proyecto aborda dos tareas complementarias:

| Tarea | Modelos | Mejor resultado |
|---|---|---|
| **Clasificación multiclase** | CNN Base, ResNet-50, EfficientNet-B0 | ResNet-50: **99.00% accuracy** |
| **Segmentación semántica** | U-Net | **Dice Score: 0.8291** |

### Clases detectadas
- 🔴 **Glioma** — tumor maligno de células gliales
- 🟠 **Meningioma** — tumor de las meninges, generalmente benigno
- 🟡 **Pituitario** — adenoma de la glándula hipófisis
- 🟢 **Sin tumor** — imagen de MRI normal

---

## 📊 Resultados

### Clasificación

| Modelo | Accuracy | F1 Macro | AUC-ROC | Parámetros |
|---|---|---|---|---|
| CNN Base (baseline) | 91.20% | 0.9171 | 0.9868 | 2.49M |
| ResNet-50 | **98.30%** | **0.9911** | **0.9985** | 24.03M |
| EfficientNet-B0 | 97.00% | 0.9721 | 0.9982 | 4.34M |

### Segmentación

| Modelo | Dice Score | IoU |
|---|---|---|
| U-Net | **0.8381** | **0.7296** |

---

## 🗂️ Estructura del repositorio

```
brain-tumor-classifier/
│
├── 📓 notebook/
│   └── brain_tumor_classifier.ipynb   # Notebook completo (Colab)
│
├── 📁 src/
│   ├── models.py                      # CNN Base, ResNet-50, EfficientNet-B0, U-Net
│   ├── dataset.py                     # Datasets de clasificación y segmentación
│   ├── train.py                       # Loops de entrenamiento
│   ├── evaluate.py                    # Métricas: F1, AUC-ROC, Dice, IoU
│   └── gradcam.py                     # Implementación Grad-CAM
│
├── 📁 results/
│   ├── eda_distribucion.png
│   ├── eda_mascaras.png
│   ├── curvas_entrenamiento.png
│   ├── matrices_confusion.png
│   ├── curvas_roc_por_clase.png
│   ├── gradcam_resnet50.png
│   ├── gradcam_efficientnet_b0.png
│   └── predicciones_unet.png
│
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🚀 Instalación y uso

### Requisitos previos
- Cuenta de Kaggle (para descargar el dataset)
- Cuenta de Google (para Google Drive y Colab)

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset

El proyecto utiliza el dataset **BRISC 2025** disponible en Kaggle:

👉 [https://www.kaggle.com/datasets/briscdataset/brisc2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)

Estructura esperada después de descargar:
```
brisc2025/
├── classification_task/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   └── test/
│       └── (mismas carpetas)
└── segmentation_task/
    ├── train/
    │   ├── images/    ← MRI en .jpg
    │   └── masks/     ← Máscaras en .png
    └── test/
        └── (mismas carpetas)
```

### 4. Ejecutar en Google Colab

1. Subir el dataset a Google Drive
2. Abrir [Google Colab](https://colab.research.google.com)
3. Subir `notebook/brain_tumor_classifier.ipynb`
4. Activar GPU: `Entorno de ejecución → Cambiar tipo → GPU T4`
5. Actualizar la variable `BRISC_DIR` con tu ruta de Drive:

```python
BRISC_DIR = '/content/drive/MyDrive/tu-carpeta/brisc2025'
```

6. Ejecutar todas las celdas en orden

---

## 🏗️ Arquitecturas implementadas

### Clasificación

**CNN Base (Baseline)**
```
Input (3×224×224)
→ Conv2D(32) → BN → ReLU → MaxPool
→ Conv2D(64) → BN → ReLU → MaxPool
→ Conv2D(128) → BN → ReLU → MaxPool
→ Conv2D(256) → BN → ReLU → MaxPool
→ AdaptiveAvgPool → Flatten
→ Linear(4096, 512) → ReLU → Dropout(0.5)
→ Linear(512, 4)
```

**ResNet-50 (Transfer Learning)**
- Pesos preentrenados: ImageNet (IMAGENET1K_V2)
- Capas congeladas: layers 1-3
- Fine-tuning: layer4 + cabeza personalizada
- Cabeza: Dropout(0.4) → Linear(2048, 256) → ReLU → Dropout(0.3) → Linear(256, 4)

**EfficientNet-B0 (Transfer Learning)**
- Pesos preentrenados: ImageNet (IMAGENET1K_V1)
- Fine-tuning: features.7, features.8 + clasificador
- Solo 33.6% de parámetros entrenables respecto a ResNet-50

### Segmentación

**U-Net**
- Encoder: 4 niveles con DoubleConv (64→128→256→512)
- Bottleneck: 1024 canales
- Decoder: 4 niveles simétricos con skip connections
- Salida: máscara binaria (1×224×224)
- Loss: Dice Loss

---

## 📈 Reproducibilidad

Todos los experimentos fueron realizados con las siguientes semillas fijas:

```python
torch.manual_seed(42)
numpy.random.seed(42)
```

Hiperparámetros utilizados:

| Parámetro | Valor |
|---|---|
| Image size | 224 × 224 |
| Batch size | 16 |
| Epochs máx. | 20 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | Paciencia = 5 épocas |
| Hardware | GPU NVIDIA Tesla T4 |

---

## 📦 requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
opencv-python>=4.8.0
tqdm>=4.65.0
```

---

## 🔥 Grad-CAM

El sistema implementa Gradient-weighted Class Activation Mapping (Grad-CAM) para visualizar las regiones de la MRI que activaron la predicción del modelo:

```python
gradcam = GradCAM(model, target_layer)
cam, pred, confianza = gradcam.generate(img_tensor)
# cam: mapa de calor normalizado [0,1]
# pred: clase predicha
# confianza: porcentaje de certeza del modelo
```

---

## 📚 Referencias

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015
2. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
3. Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019
4. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017
5. Sajjad et al., "Multi-Grade Brain Tumor Classification Using Deep CNN," Journal of Computational Science 2019

---

## 📄 Licencia

MIT License — libre para uso académico y educativo.
