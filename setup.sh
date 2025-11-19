#!/bin/bash
# Setup automático para Butterfly Preprocessing
# Instala OpenCV e todas as dependências necessárias

set -e  # Para na primeira falha

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  🦋 Setup Automático${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Detecta sistema operacional
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
else
    OS=$(uname -s)
fi

echo -e "${BLUE}Sistema detectado: ${OS}${NC}"
echo ""

# 1. Atualizar sistema
echo -e "${YELLOW}[1/3] Atualizando sistema...${NC}"
sudo apt-get update -qq

# 2. Instalar dependências
echo -e "${YELLOW}[2/3] Instalando dependências...${NC}"
echo "  - Build essentials"
echo "  - CMake"
echo "  - OpenCV"
echo ""

sudo apt-get install -y -qq \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    python3-opencv \
    > /dev/null 2>&1

echo -e "${GREEN}✓ Dependências do sistema instaladas${NC}"
echo ""

# 3. Instalar pacotes Python
echo -e "${YELLOW}[3/3] Instalando pacotes Python...${NC}"
pip install -q --upgrade pip
pip install -q kaggle python-dotenv

echo -e "${GREEN}✓ Pacotes Python instalados${NC}"
echo ""

# Verificar instalação do OpenCV
echo -e "${BLUE}Verificando OpenCV...${NC}"
OPENCV_VERSION=$(pkg-config --modversion opencv4 2>/dev/null || echo "não encontrado")
if [ "$OPENCV_VERSION" != "não encontrado" ]; then
    echo -e "${GREEN}✓ OpenCV ${OPENCV_VERSION} instalado com sucesso${NC}"
else
    echo -e "${RED}⚠ Aviso: OpenCV pode não estar configurado corretamente${NC}"
fi
echo ""

# Instruções finais
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ SETUP CONCLUÍDO!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Próximos passos:${NC}"
echo ""
echo "1. Configure suas credenciais Kaggle:"
echo "   export KAGGLE_USERNAME='seu_username'"
echo "   export KAGGLE_KEY='sua_key'"
echo ""
echo "2. Execute o pipeline completo:"
echo "   make all-in-one"
echo ""
echo "Ou execute separadamente:"
echo "   make download    # Baixar dataset"
echo "   make compile     # Compilar código"
echo "   make preprocess  # Processar imagens"
echo ""