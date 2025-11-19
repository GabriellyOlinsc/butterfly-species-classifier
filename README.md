# 🦋 Butterfly Species Classifier  

This repository focuses on developing a classical image processing system for **butterfly species classification** using C++ and OpenCV. The project follows the recommended academic pipeline, including preprocessing, segmentation, feature extraction, and classification.

## Objective

Develop a complete pipeline capable of automatically identifying butterfly species from photographs using classical computer vision and machine learning methods.

---

## Dataset Used
**Butterfly Image Classification Dataset**  
https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

## Tecnologias Utilizadas

- **Linguagem:** C++  
- **Bibliotecas principais:**  
  - OpenCV (pré-processamento, segmentação, descritores, classificadores)  
  - (Opcional) dlib ou implementação própria para LBP

## Technologies Used
- **Language:** C++  
- **Main Libraries:**  
  - OpenCV (preprocessing, segmentation, descriptors, classifiers)  
  - (Optional) dlib or a custom implementation for LBP

---

## How to Run
```bash
mkdir build
cd build

cmake ..
make

./preprocess_butterflies <pasta_entrada> <pasta_saida> [arquivo_metricas.csv]
```

## Examples
```bash
# Process training images
./preprocess_butterflies ../dataset/train ../preprocessed/train metrics_train.csv

# Process training images
./preprocess_butterflies ../dataset/val ../preprocessed/val metrics_val.csv
```
---

## Folder Structure
```bash
butterfly-species-classifier/
│
├── data/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── descriptors/
│   ├── classification/
│   └── main.cpp
│
├── docs/
│   ├── report/
│   └── slides/
│
├── results/
│   ├── metrics/
│   └── confusion_matrix/
│
└── README.md
```
