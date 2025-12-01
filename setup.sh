#!/bin/bash
# Setup automático para Butterfly Preprocessing
# Instala OpenCV, OpenMP e todas as dependências necessárias

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  🦋 Setup Automático (Otimizado)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
else
    OS=$(uname -s)
fi

echo -e "${BLUE}Sistema: ${OS}${NC}"
echo ""

echo -e "${YELLOW}[1/3] Atualizando sistema...${NC}"
sudo apt-get update -qq

echo -e "${YELLOW}[2/3] Instalando dependências...${NC}"
echo "  - Build essentials"
echo "  - CMake"
echo "  - OpenCV"
echo "  - OpenMP (paralelização)"
echo ""

sudo apt-get install -y -qq \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    python3-opencv \
    libomp-dev \
    > /dev/null 2>&1

echo -e "${GREEN}✓ Dependências instaladas${NC}"
echo ""

echo -e "${YELLOW}[3/3] Instalando pacotes Python...${NC}"
pip install -q --upgrade pip
pip install -q kaggle python-dotenv numpy pandas scikit-learn opencv-python matplotlib seaborn joblib

echo -e "${GREEN}✓ Pacotes Python instalados${NC}"
echo ""

echo -e "${BLUE}Verificando instalações...${NC}"

OPENCV_VERSION=$(pkg-config --modversion opencv4 2>/dev/null || echo "não encontrado")
if [ "$OPENCV_VERSION" != "não encontrado" ]; then
    echo -e "${GREEN}✓ OpenCV ${OPENCV_VERSION}${NC}"
else
    echo -e "${RED}⚠ OpenCV não configurado${NC}"
fi

if echo | gcc -fopenmp -x c - -o /dev/null 2>/dev/null; then
    echo -e "${GREEN}✓ OpenMP disponível${NC}"
else
    echo -e "${YELLOW}⚠ OpenMP não detectado${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ SETUP CONCLUÍDO!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Próximos passos:${NC}"
echo ""
echo "1. Configure credenciais Kaggle:"
echo "   export KAGGLE_USERNAME='seu_username'"
echo "   export KAGGLE_KEY='sua_key'"
echo ""
echo "2. Execute o pipeline:"
echo "   make full-pipeline"
echo ""
echo -e "${CYAN}Com OpenMP: Features em ~5-8min (vs 40min)${NC}"
echo ""