#!/bin/bash
# Script de setup rápido para GitHub Codespaces

set -e  # Para na primeira falha

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  🦋 Setup - Butterfly Preprocessing${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 1. Verificar se está no Codespaces
if [ -n "$CODESPACES" ]; then
    echo -e "${GREEN}✓ Rodando no GitHub Codespaces${NC}"
else
    echo -e "${YELLOW}⚠ Não está no Codespaces (mas pode funcionar)${NC}"
fi
echo ""

# 2. Instalar dependências do sistema
echo -e "${BLUE}[1/5] Instalando dependências do sistema...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake pkg-config libopencv-dev > /dev/null 2>&1
echo -e "${GREEN}✓ Dependências instaladas${NC}"
echo ""

# 3. Instalar dependências Python
echo -e "${BLUE}[2/5] Instalando dependências Python...${NC}"
pip install -q --upgrade pip
pip install -q kaggle opendatasets
echo -e "${GREEN}✓ Python packages instalados${NC}"
echo ""

# 4. Verificar credenciais
echo -e "${BLUE}[3/5] Verificando credenciais do Kaggle...${NC}"

# Tenta carregar do .env se existir
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ Arquivo .env encontrado${NC}"
    export $(cat .env | grep -v '^#' | xargs)
fi

if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo -e "${GREEN}✓ Credenciais encontradas${NC}"
    if [ -f ".env" ]; then
        echo -e "  Origem: arquivo .env"
    else
        echo -e "  Origem: variáveis de ambiente"
    fi
    echo -e "  Username: ${KAGGLE_USERNAME}"
else
    echo -e "${YELLOW}⚠ Credenciais não encontradas${NC}"
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Configure suas credenciais do Kaggle${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
    echo ""
    echo "1. Acesse: https://www.kaggle.com/settings"
    echo "2. Role até 'API' e clique em 'Create New API Token'"
    echo "3. Abra o arquivo kaggle.json baixado"
    echo ""
    
    read -p "Deseja fornecer as credenciais agora? (s/n): " resposta
    
    if [ "$resposta" = "s" ] || [ "$resposta" = "S" ]; then
        echo ""
        read -p "Kaggle Username: " username
        read -p "Kaggle Key: " key
        
        if [ -n "$username" ] && [ -n "$key" ]; then
            export KAGGLE_USERNAME="$username"
            export KAGGLE_KEY="$key"
            
            # Pergunta se quer salvar no .env
            read -p "Deseja salvar no arquivo .env? (s/n): " save_env
            
            if [ "$save_env" = "s" ] || [ "$save_env" = "S" ]; then
                cat > .env << EOF
# Credenciais da API do Kaggle
KAGGLE_USERNAME=$username
KAGGLE_KEY=$key
EOF
                echo -e "${GREEN}✓ Credenciais salvas em .env${NC}"
                echo -e "${YELLOW}⚠ IMPORTANTE: Não commite o arquivo .env no Git!${NC}"
            else
                # Adiciona ao .bashrc para persistir na sessão
                if ! grep -q "KAGGLE_USERNAME" ~/.bashrc; then
                    echo "" >> ~/.bashrc
                    echo "# Kaggle API Credentials" >> ~/.bashrc
                    echo "export KAGGLE_USERNAME='$username'" >> ~/.bashrc
                    echo "export KAGGLE_KEY='$key'" >> ~/.bashrc
                    echo -e "${GREEN}✓ Credenciais salvas em ~/.bashrc${NC}"
                fi
            fi
            
            echo -e "${GREEN}✓ Credenciais configuradas!${NC}"
        else
            echo -e "${RED}✗ Credenciais inválidas${NC}"
            exit 1
        fi
    else
        echo ""
        echo -e "${YELLOW}Configure manualmente:${NC}"
        echo ""
        echo "  Opção A - Arquivo .env (recomendado):"
        echo "    cp .env.example .env"
        echo "    # Edite .env com suas credenciais"
        echo ""
        echo "  Opção B - Variáveis de ambiente:"
        echo "    export KAGGLE_USERNAME='seu_username'"
        echo "    export KAGGLE_KEY='sua_key'"
        echo ""
        echo "  Opção C - Secrets do Codespaces:"
        echo "    Settings > Secrets and variables > Codespaces"
        exit 1
    fi
fi
echo ""

# 5. Baixar dataset (opcional)
echo -e "${BLUE}[4/5] Dataset${NC}"
if [ -d "dataset/train" ]; then
    echo -e "${GREEN}✓ Dataset já existe, pulando download${NC}"
else
    echo "O dataset de borboletas tem ~1-2GB"
    read -p "Deseja baixar agora? (s/n): " download_now
    
    if [ "$download_now" = "s" ] || [ "$download_now" = "S" ]; then
        echo -e "${YELLOW}⏳ Baixando dataset... (pode levar 5-10 min)${NC}"
        python3 download_dataset.py
        echo -e "${GREEN}✓ Dataset baixado!${NC}"
    else
        echo -e "${YELLOW}⊘ Download pulado. Execute depois: make download${NC}"
    fi
fi
echo ""

# 6. Compilar código C++
echo -e "${BLUE}[5/5] Compilando código C++...${NC}"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake .. > /dev/null
make
cd ..
echo -e "${GREEN}✓ Código compilado!${NC}"
echo ""

# Resumo final
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ SETUP CONCLUÍDO!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "📋 Próximos passos:"
echo ""

if [ ! -d "dataset/train" ]; then
    echo "1. Baixar dataset:"
    echo "   make download"
    echo ""
    echo "2. Processar imagens:"
else
    echo "1. Processar imagens:"
fi

echo "   make preprocess"
echo ""
echo "OU executar tudo de uma vez:"
echo "   make quick"
echo ""
echo "📖 Para mais informações: SETUP-CODESPACES.md"
echo ""