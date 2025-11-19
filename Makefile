# Makefile para Pipeline de Pré-processamento de Borboletas
# Automatiza: download do dataset, compilação e execução

.PHONY: all setup download compile preprocess clean help

# Configurações
DATASET_DIR = dataset
PREPROCESSED_DIR = preprocessed
BUILD_DIR = build
PYTHON = python3

# Cores para output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

all: help

help:
	@echo "$(GREEN)================================================$(NC)"
	@echo "$(GREEN)  Pipeline de Pré-processamento - Borboletas$(NC)"
	@echo "$(GREEN)================================================$(NC)"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  $(YELLOW)make setup$(NC)            - Instala dependências Python"
	@echo "  $(YELLOW)make check-credentials$(NC) - Verifica se credenciais estão configuradas"
	@echo "  $(YELLOW)make download$(NC)         - Baixa dataset do Kaggle"
	@echo "  $(YELLOW)make compile$(NC)          - Compila código C++"
	@echo "  $(YELLOW)make preprocess$(NC)       - Executa pré-processamento completo"
	@echo "  $(YELLOW)make quick$(NC)            - Download + Compile + Preprocess (tudo)"
	@echo "  $(YELLOW)make stats$(NC)            - Mostra estatísticas do dataset"
	@echo "  $(YELLOW)make clean$(NC)            - Remove arquivos temporários"
	@echo "  $(YELLOW)make clean-all$(NC)        - Remove tudo (inclusive dataset)"
	@echo ""
	@echo "$(YELLOW)📖 Para uso no Codespaces, leia: SETUP-CODESPACES.md$(NC)"
	@echo ""
	@echo "Uso rápido no Codespaces:"
	@echo "  1. Configure credenciais:"
	@echo "     $ export KAGGLE_USERNAME='seu_username'"
	@echo "     $ export KAGGLE_KEY='sua_key'"
	@echo "  2. Execute tudo:"
	@echo "     $ make setup"
	@echo "     $ make quick"
	@echo ""

# Instala dependências Python
setup:
	@echo "$(GREEN)=== Instalando dependências ===$(NC)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install kaggle opendatasets python-dotenv
	@echo "$(GREEN)✓ Dependências Python instaladas$(NC)"
	@echo ""
	@echo "$(YELLOW)PRÓXIMO PASSO:$(NC) Configure as credenciais do Kaggle"
	@echo ""
	@echo "  🏠 USO LOCAL (recomendado):"
	@echo "     1. Copie o arquivo de exemplo:"
	@echo "        cp .env.example .env"
	@echo "     2. Edite .env e adicione suas credenciais"
	@echo ""
	@echo "  ☁️  CODESPACES:"
	@echo "     Configure secrets no repositório ou use:"
	@echo "     export KAGGLE_USERNAME='seu_username'"
	@echo "     export KAGGLE_KEY='sua_key'"
	@echo ""
	@echo "  Obtenha suas credenciais em: https://www.kaggle.com/settings"
	@echo ""

# Verifica se credenciais estão configuradas
check-credentials:
	@echo "$(GREEN)=== Verificando credenciais ===$(NC)"
	@if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then \
		echo "$(RED)✗ Credenciais não configuradas!$(NC)"; \
		echo ""; \
		echo "Configure as variáveis de ambiente:"; \
		echo "  export KAGGLE_USERNAME='seu_username'"; \
		echo "  export KAGGLE_KEY='sua_key'"; \
		echo ""; \
		echo "Ou veja o guia: SETUP-CODESPACES.md"; \
		exit 1; \
	else \
		echo "$(GREEN)✓ Credenciais configuradas$(NC)"; \
		echo "  Username: $KAGGLE_USERNAME"; \
	fi

# Baixa dataset do Kaggle
download: check-credentials
	@echo "$(GREEN)=== Baixando dataset do Kaggle ===$(NC)"
	@if [ ! -f download_dataset.py ]; then \
		echo "$(RED)✗ Arquivo download_dataset.py não encontrado!$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) download_dataset.py
	@echo "$(GREEN)✓ Dataset baixado e organizado$(NC)"

# Compila código C++
compile:
	@echo "$(GREEN)=== Compilando código C++ ===$(NC)"
	@if [ ! -d "$(BUILD_DIR)" ]; then mkdir $(BUILD_DIR); fi
	cd $(BUILD_DIR) && cmake .. && make
	@echo "$(GREEN)✓ Compilação concluída$(NC)"

# Executa pré-processamento
preprocess: compile
	@echo "$(GREEN)=== Executando pré-processamento ===$(NC)"
	@if [ ! -d "$(DATASET_DIR)/train" ]; then \
		echo "$(RED)✗ Dataset não encontrado!$(NC)"; \
		echo "Execute: make download"; \
		exit 1; \
	fi
	
	@mkdir -p $(PREPROCESSED_DIR)/train
	@mkdir -p $(PREPROCESSED_DIR)/test
	@if [ -d "$(DATASET_DIR)/val" ]; then mkdir -p $(PREPROCESSED_DIR)/val; fi
	
	@echo "$(YELLOW)Processando imagens de treino...$(NC)"
	./$(BUILD_DIR)/preprocess_butterflies \
		$(DATASET_DIR)/train \
		$(PREPROCESSED_DIR)/train \
		metrics_train.csv
	
	@echo ""
	@echo "$(YELLOW)Processando imagens de teste...$(NC)"
	./$(BUILD_DIR)/preprocess_butterflies \
		$(DATASET_DIR)/test \
		$(PREPROCESSED_DIR)/test \
		metrics_test.csv
	
	@if [ -d "$(DATASET_DIR)/val" ]; then \
		echo ""; \
		echo "$(YELLOW)Processando imagens de validação...$(NC)"; \
		./$(BUILD_DIR)/preprocess_butterflies \
			$(DATASET_DIR)/val \
			$(PREPROCESSED_DIR)/val \
			metrics_val.csv; \
	fi
	
	@echo ""
	@echo "$(GREEN)✓ Pré-processamento concluído!$(NC)"
	@echo ""
	@echo "Resultados salvos em:"
	@echo "  - $(PREPROCESSED_DIR)/"
	@echo "  - metrics_*.csv"

# Execução rápida: tudo de uma vez
quick: download compile preprocess
	@echo ""
	@echo "$(GREEN)================================================$(NC)"
	@echo "$(GREEN)  ✓ PIPELINE COMPLETO EXECUTADO!$(NC)"
	@echo "$(GREEN)================================================$(NC)"

# Limpa arquivos temporários
clean:
	@echo "$(YELLOW)Removendo arquivos temporários...$(NC)"
	rm -rf $(BUILD_DIR)
	rm -f metrics_*.csv
	@echo "$(GREEN)✓ Limpeza concluída$(NC)"

# Limpa tudo (inclusive dataset)
clean-all: clean
	@echo "$(RED)Removendo dataset e imagens processadas...$(NC)"
	rm -rf $(DATASET_DIR)
	rm -rf $(PREPROCESSED_DIR)
	rm -rf dataset_temp
	@echo "$(GREEN)✓ Limpeza completa realizada$(NC)"

# Mostra estatísticas do dataset
stats:
	@echo "$(GREEN)=== Estatísticas do Dataset ===$(NC)"
	@if [ -d "$(DATASET_DIR)" ]; then \
		echo ""; \
		for split in train test val; do \
			if [ -d "$(DATASET_DIR)/$$split" ]; then \
				echo "$(YELLOW)$$split/:$(NC)"; \
				n_species=$$(ls -d $(DATASET_DIR)/$$split/*/ 2>/dev/null | wc -l); \
				n_images=$$(find $(DATASET_DIR)/$$split -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l); \
				echo "  Espécies: $$n_species"; \
				echo "  Imagens: $$n_images"; \
				echo ""; \
			fi \
		done \
	else \
		echo "$(RED)Dataset não encontrado. Execute: make download$(NC)"; \
	fi

# Mostra informações do sistema
info:
	@echo "$(GREEN)=== Informações do Sistema ===$(NC)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo "CMake: $$(cmake --version | head -n1)"
	@echo "OpenCV: $$(pkg-config --modversion opencv4 2>/dev/null || echo 'não detectado via pkg-config')"
	@echo "GCC: $$(gcc --version | head -n1)"
	@echo ""