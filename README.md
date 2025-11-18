# 🦋 Butterfly Species Classifier  

Repositório dedicado ao desenvolvimento de um sistema clássico de processamento de imagens para **classificação de espécies de borboletas** utilizando C++ e OpenCV. O projeto segue o pipeline recomendado pela disciplina, incluindo pré-processamento, segmentação, extração de descritores e classificação.


## Objetivo

Desenvolver um pipeline completo capaz de identificar automaticamente a espécie de uma borboleta a partir de uma fotografia, utilizando exclusivamente métodos clássicos de visão computacional e aprendizado de máquina.

---

## Dataset Utilizado
**Butterfly Image Classification Dataset**  
https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

- ~7.000 imagens  
- Classificação multi-classe  
- Apenas rótulos de espécie (sem máscaras ou bounding boxes)  
- Resolução variada (~224×224 px em média)

A divisão utilizada será:
- 70% treino  
- 15% validação  
- 15% teste  
Com **random seed fixa** para garantir reprodutibilidade.


## Tecnologias Utilizadas

- **Linguagem:** C++  
- **Bibliotecas principais:**  
  - OpenCV (pré-processamento, segmentação, descritores, classificadores)  
  - (Opcional) dlib ou implementação própria para LBP


## Como rodar
```bash
mkdir build
cd build
cmake ..
make
./butterfly_classifier
```

## Estrutura de pasta:
```bash
butterfly-species-classifier/
│
├── data/
│ ├── raw/
│ ├── train/
│ ├── val/
│ └── test/
│
├── src/
│ ├── preprocessing/
│ ├── segmentation/
│ ├── descriptors/
│ ├── classification/
│ └── main.cpp
│
├── docs/
│ ├── relatório/
│ └── slides/
│
├── results/
│ ├── metrics/
│ └── confusion_matrix/
│
└── README.md
```



