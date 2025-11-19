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

## Quick Use
This should be used on the first execution (complete setup)
```bash
# 1. Install OpenCV and dependencies (apenas uma vez)
make setup-system

# 2. Configure Kaggle credentials
export KAGGLE_USERNAME='seu_username'
export KAGGLE_KEY='sua_key'

# 3. Execute complete pipeline
make all-in-one
```

## How to Run
After the first execution, there's no need to run everything all over again
```bash
# Process only images (dataset already exists)
make preprocess

# If only C++ were altered
make recompile
make preprocess

# Clean environment and run
make clean-preprocessed
make preprocess
```

## Folder Structure
```bash
butterfly-species-classifier/
├── Makefile              # Automação 
├── preprocessing.cpp     # Código principal C++
├── download_dataset.py   # Script de download
├── CMakeLists.txt        # Configuração CMake
└── README.md
```
