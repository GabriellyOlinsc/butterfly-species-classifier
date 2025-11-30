# 🦋  Classificador de Espécies de Borboletas

Este repositório apresenta o desenvolvimento de um sistema clássico de processamento de imagens para **classificação de espécies de borboletas** utilizando C++ e OpenCV. O projeto segue o pipeline acadêmico recomendado, incluindo pré-processamento, segmentação, extração de características e classificação.

## Objetivo
Desenvolver um pipeline completo capaz de identificar automaticamente espécies de borboletas a partir de fotografias, utilizando métodos tradicionais de visão computacional e aprendizado de máquina.

---

## Base de Dados Utilizada

**Butterfly Image Classification Dataset**  
Disponível em:  
https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

## Tecnologias Utilizadas

- **Linguagem:** C++ e python
- **Bibliotecas principais:**  
  - OpenCV (pré-processamento, segmentação, descritores, classificadores)  
  - (Opcional) dlib ou implementação própria para LBP

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
make all-in-one
```

## Como Executar (após a primeira vez)

Após o ambiente estar configurado, você não precisa repetir toda a instalação:
```bash
# Processar apenas as imagens (dataset já existe)
make preprocess

# Caso apenas o código C++ tenha sido alterado
make recompile
make preprocess

# Limpar ambiente e rodar do zero
make clean
make preprocess
```

## Estrutura de Pastas
```bash
butterfly-classification/
├── dataset/                    # Imagens (baixadas do Kaggle)
├── models/                     # Modelos treinados (.pkl)
├── evaluation_results/         # Gráficos e relatórios
├── preprocessing.cpp           # Pré-processamento (C++)
├── feature_extraction.cpp      # HOG + LBP (C++)
├── train_classifier.py         # SVM + Random Forest
├── evaluate_model.py           # Análise de erro
├── Makefile                    # Automação
└── README.md
```
