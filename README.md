# 🦋  Classificador de Espécies de Borboletas

Este repositório apresenta o desenvolvimento de um sistema clássico de processamento de imagens para **classificação de espécies de borboletas** utilizando C++ e OpenCV. O projeto segue o pipeline acadêmico recomendado, incluindo pré-processamento, segmentação, extração de características e classificação.

## Objetivo
Desenvolver um pipeline completo capaz de identificar automaticamente espécies de borboletas a partir de fotografias, utilizando métodos tradicionais de visão computacional e aprendizado de máquina.
 - Pré-processar imagens (C++ / OpenCV)
 - Extrair características HOG + LBP + Cor (C++ com OpenMP)
 - Treinar classificadores tradicionais (Python / scikit-learn)
 - Avaliar o desempenho final
 - Realizar predições individuais ou em lote
---

## Base de Dados Utilizada

**Butterfly Image Classification Dataset**  
Disponível em:  
https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

## Tecnologias Utilizadas

- **Linguagem:** C++ e python
- **Bibliotecas principais:**  
  - OpenCV (pré-processamento, segmentação, descritores, classificadores)  
  - OpenMp (Paralelização da extração de features)
  - scikit-learn — SVM, Logistic Regression, Random Forest

---
##  Configuração Inicial – Kaggle Dataset

Para baixar o dataset automaticamente, é necessário configurar as credenciais da API do Kaggle.

### **Passo a passo:**
1. Acesse sua conta Kaggle:  
   https://www.kaggle.com/settings/account
2. Vá até a seção **API**
3. Clique em **Create New Token**
4. Baixe e abra o arquivo **kaggle.json**

Exemplo de conteúdo:

```json
  {
     "username": "seu_username_aqui",
     "key": "sua_chave_longa_aqui123456789"
  }
```

## Uso Rápido

Use estes comandos na primeira execução:

```bash
# 1. Instalar OpenCV e dependências (apenas uma vez)
make setup-system

# 2. Configurar credenciais do Kaggle
export KAGGLE_USERNAME='seu_username'
export KAGGLE_KEY='sua_key'

# 3. Executar pipeline completo
make full-pipeline
```

## Comandos Principais

Após o ambiente estar configurado, você não precisa repetir toda a instalação:
```bash
#Setup inicial
make setup-system     # Instala OpenCV / verifica OpenMP
make setup            # Instala dependências Python
make download         # Baixa dataset do Kaggle

#Compilação e pipeline
make compile          # Compila C++ com -O3 e OpenMP
make features         # Extrai features (paralelo)
make train            # Treina SVM+LR+RandomForest
make evaluate         # Avalia modelos```

#Pipeline completo
make pipeline         # compile → features → train → evaluate
make full-pipeline    # setup + download + pipeline
```

**Predições**
```bash
make predict-one IMAGE=dataset/train/Image_1.jpg      #testa uma única imagem
make evaluate-prediction                              #testa toda a pasta dataset/train 
```
**Limpeza**
```bash
make clean            # limpa build/
make clean-all        # remove dataset, modelos e features
```

## Estrutura de Pastas
```bash
butterfly-classification/
├── dataset/                  # Base Kaggle (train/test)
├── preprocessed/             # Imagens pré-processadas (C++)
├── build/                    # Binários C++ compilados
├── models/                   # Modelos .pkl treinados
├── evaluation_results/       # Resultados e gráficos
├── features_combined.csv     # Features geradas (HOG+LBP+Cor)
├── download_dataset.py       # Kaggle downloader
├── preprocessing.cpp           # Pré-processamento (C++)
├── feature_extraction.cpp      # HOG + LBP (C++)
├── train_classifier.py         # SVM + Random Forest
├── predict_butterfly.py      
├── evaluate_model.py           # Análise de erro
├── Makefile                    # Automação
├── CMakeLists                  
└── README.md
```
