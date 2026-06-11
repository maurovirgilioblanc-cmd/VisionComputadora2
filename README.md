# 🧠 Clasificación y Segmentación de Tumores Cerebrales en MRI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20T4-20BEFF?logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green)

Trabajo Práctico Final — Visión por Computadora  
**Autores:** Mauro Virgilio Blanc · Juan Pablo Imbrigno · Andrea Ferenaz  
**Año:** 2026

---

## 📋 Descripción

Sistema de diagnóstico asistido por inteligencia artificial para la detección y segmentación de tumores cerebrales en imágenes de resonancia magnética (MRI). El proyecto aborda dos tareas complementarias:

| Tarea | Modelos | Mejor resultado |
|---|---|---|
| **Clasificación multiclase** | CNN Base (96.40% accuracy, ResNet-50 (98.70% accuracy), EfficientNet-B0 (97.50% accuracy) | ResNet-50: **98.70% accuracy** |
| **Segmentación semántica** | U-Net | **Dice Score: 0.8434** |

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
| CNN Base (baseline) | 96.40% | 0.9666 | 0.9966 | 2.49M |
| ResNet-50 | **98.70%** | **0.9880** | **0.9996** | 24.03M |
| EfficientNet-B0 | 97.50% | 0.9754 | 0.9991 | 4.34M |

### Segmentación

| Modelo | Dice Score | IoU |
|---|---|---|
| U-Net | **0.8434** | **0.7220** |

---

## 🗂️ Estructura del repositorio

```
brain-tumor-classifier/
│
├── 📓 notebook/
│   └── brain_tumor_classifier.ipynb     # Notebook completo (Kaggle)
│
├── 📁 results/
│   ├── eda_distribucion.png             # Distribución de clases
│   ├── eda_mascaras.png                 # Pares imagen/máscara
│   ├── curvas_entrenamiento.png         # Loss y accuracy por época
│   ├── matrices_confusion.png          # Matrices normalizadas x3
│   ├── curvas_roc_por_clase.png        # AUC-ROC x3 modelos
│   ├── curvas_roc_comparativa.png      # ROC superpuesto
│   ├── gradcam_resnet50.png            # Grad-CAM ResNet
│   ├── gradcam_efficientnet_b0.png     # Grad-CAM EfficientNet
│   └── predicciones_unet.png          # Segmentación U-Net
│
├── 📄 README.md
└── 📄 requirements.txt
```

> ⚠️ Los checkpoints (`.pth`) no están incluidos en este repositorio debido a su tamaño (10–124 MB). Ver sección de descarga abajo.

---

## 🚀 Instalación y uso

### Requisitos previos
- Cuenta de Kaggle (para el dataset y ejecutar el notebook)

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset y checkpoints

**Dataset BRISC 2025(Entrenamiento y Evaluación):**

👉 [https://www.kaggle.com/datasets/briscdataset/brisc2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)

**Dataset Brain Tumor MRI Dataset (Validación):**

👉 [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Checkpoints de los modelos entrenados:**

👉 [https://www.kaggle.com/tu-usuario/brain-tumor-classifier/output](https://www.kaggle.com/tu-usuario/brain-tumor-classifier/output)

Descargá los siguientes archivos y colocálos en una carpeta `models/`:

| Archivo | Tamaño | Descripción |
|---|---|---|
| `cls_cnn_base.pth` | 10 MB | CNN Base (baseline) |
| `cls_resnet50.pth` | 96 MB | ResNet-50 fine-tuned |
| `cls_efficientnet_b0.pth` | 17 MB | EfficientNet-B0 fine-tuned |
| `seg_unet.pth` | 124 MB | U-Net segmentación |

Estructura esperada del dataset BRISC 2025 después de descargar:
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

Estructura esperada del dataset Brain Tumor MRI Dataset después de descargar:
```
/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Training/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 4. Ejecutar en Kaggle

1. Ir a [kaggle.com/code](https://kaggle.com/code) 
2. Agregar el dataset BRISC 2025 como input
3. Agregar el dataset Brain Tumor MRI Dataset como input
4. Subir `notebook/brain_tumor_classifier.ipynb`
5. Activar GPU: `Settings → Accelerator → GPU T4`
6. Ejecutar todas las celdas en orden

---

## ⚠️ Notas sobre los modelos

Los checkpoints (`.pth`) **no están incluidos en este repositorio** debido a su tamaño (10–124 MB cada uno). Están disponibles para descarga directa desde la salida del notebook en Kaggle (ver link arriba).

Para reproducir el entrenamiento completo desde cero, ejecutar el notebook en Kaggle con GPU T4 activada. El tiempo estimado de entrenamiento es de **3–4 horas** para los 4 modelos completos.

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
| Hardware | GPU NVIDIA Tesla T4 (Kaggle) |

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
