# Makefile Simplificado - Butterfly Preprocessing

.PHONY: all setup-system setup download compile preprocess clean help

# Diretórios
DATASET_DIR = dataset
PREPROCESSED_DIR = preprocessed
BUILD_DIR = build

# Cores para output
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m

all: help

help:
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)  Butterfly Image Preprocessing$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "Comandos:"
	@echo "  $(YELLOW)make setup-system$(NC) - Instala OpenCV e dependências (primeira vez)"
	@echo "  $(YELLOW)make setup$(NC)        - Instala pacotes Python"
	@echo "  $(YELLOW)make download$(NC)     - Baixa dataset (requer credenciais Kaggle)"
	@echo "  $(YELLOW)make compile$(NC)      - Compila código C++"
	@echo "  $(YELLOW)make preprocess$(NC)   - Processa imagens"
	@echo "  $(YELLOW)make all-in-one$(NC)   - Faz tudo (setup + download + compile + preprocess)"
	@echo "  $(YELLOW)make clean$(NC)        - Remove arquivos temporários"
	@echo ""
	@echo "$(RED)PRIMEIRA VEZ? Execute:$(NC)"
	@echo "  $(YELLOW)make setup-system$(NC)"
	@echo ""
	@echo "Uso rápido depois do setup:"
	@echo "  1. Configure credenciais Kaggle:"
	@echo "     export KAGGLE_USERNAME='seu_username'"
	@echo "     export KAGGLE_KEY='sua_key'"
	@echo "  2. Execute: make all-in-one"
	@echo ""

setup-system:
	@if [ ! -f setup.sh ]; then \
		echo "$(RED)✗ Arquivo setup.sh não encontrado!$(NC)"; \
		exit 1; \
	fi
	@chmod +x setup.sh
	@./setup.sh

setup:
	@echo "$(GREEN)=== Instalando pacotes Python ===$(NC)"
	@python3 -m pip install -q --upgrade pip
	@python3 -m pip install -q kaggle python-dotenv
	@echo "$(GREEN)✓ Python packages instalados$(NC)"
	@echo ""
	@echo "$(YELLOW)Configure suas credenciais:$(NC)"
	@echo "  export KAGGLE_USERNAME='seu_username'"
	@echo "  export KAGGLE_KEY='sua_key'"
	@echo ""
	@echo "Obtenha em: https://www.kaggle.com/settings"

download:
	@if [ -z "$$KAGGLE_USERNAME" ] || [ -z "$$KAGGLE_KEY" ]; then \
		echo "$(RED)✗ Configure credenciais primeiro!$(NC)"; \
		echo "  export KAGGLE_USERNAME='seu_username'"; \
		echo "  export KAGGLE_KEY='sua_key'"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Baixando dataset ===$(NC)"
	@python3 download_dataset.py
	@echo "$(GREEN)✓ Download concluído$(NC)"

compile:
	@echo "$(GREEN)=== Compilando ===$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. > /dev/null && make
	@echo "$(GREEN)✓ Compilado: $(BUILD_DIR)/preprocess$(NC)"

preprocess: compile
	@if [ ! -d "$(DATASET_DIR)/train" ]; then \
		echo "$(RED)✗ Dataset não encontrado!$(NC)"; \
		echo "Execute: make download"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Pré-processamento ===$(NC)"
	@mkdir -p $(PREPROCESSED_DIR)
	@echo "$(YELLOW)[1/2] Processando treino...$(NC)"
	@./$(BUILD_DIR)/preprocess $(DATASET_DIR)/train $(PREPROCESSED_DIR)/train
	@echo ""
	@echo "$(YELLOW)[2/2] Processando teste...$(NC)"
	@./$(BUILD_DIR)/preprocess $(DATASET_DIR)/test $(PREPROCESSED_DIR)/test
	@if [ -d "$(DATASET_DIR)/val" ]; then \
		echo ""; \
		echo "$(YELLOW)[3/3] Processando validação...$(NC)"; \
		./$(BUILD_DIR)/preprocess $(DATASET_DIR)/val $(PREPROCESSED_DIR)/val; \
	fi
	@echo ""
	@echo "$(GREEN)✓ Concluído! Imagens em: $(PREPROCESSED_DIR)/$(NC)"

all-in-one: setup download compile preprocess
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)  ✓ PIPELINE COMPLETO!$(NC)"
	@echo "$(GREEN)========================================$(NC)"

clean:
	@echo "$(YELLOW)Limpando arquivos temporários...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Limpo$(NC)"

clean-all: clean
	@echo "$(RED)Removendo dataset e preprocessed...$(NC)"
	@rm -rf $(DATASET_DIR) $(PREPROCESSED_DIR)
	@echo "$(GREEN)✓ Tudo removido$(NC)"