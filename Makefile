# Makefile - Butterfly Image Classification Pipeline

.PHONY: all setup-system setup download compile preprocess features train evaluate ablation slides clean help

# Diretórios
DATASET_DIR = dataset
PREPROCESSED_DIR = preprocessed
BUILD_DIR = build
MODELS_DIR = models
RESULTS_DIR = evaluation_results

# Cores para output
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
BLUE = \033[0;34m
CYAN = \033[0;36m
NC = \033[0m

all: help

help:
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)  🦋 Butterfly Classification Pipeline$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(BLUE)SETUP INICIAL (primeira vez):$(NC)"
	@echo "  $(YELLOW)make setup-system$(NC)    - Instala OpenCV e dependências"
	@echo "  $(YELLOW)make setup$(NC)           - Instala pacotes Python"
	@echo "  $(YELLOW)make download$(NC)        - Baixa dataset do Kaggle"
	@echo ""
	@echo "$(BLUE)PIPELINE PRINCIPAL:$(NC)"
	@echo "  $(YELLOW)make compile$(NC)         - Compila código C++"
	@echo "  $(YELLOW)make preprocess$(NC)      - Pré-processa imagens"
	@echo "  $(YELLOW)make features$(NC)        - Extrai features (HOG+LBP)"
	@echo "  $(YELLOW)make train$(NC)           - Treina classificadores (SVM, RF)"
	@echo "  $(YELLOW)make evaluate$(NC)        - Avalia modelos e gera gráficos"
	@echo "  $(YELLOW)make slides$(NC)          - Gera apresentação em Markdown"
	@echo ""
	@echo "$(BLUE)ATALHOS:$(NC)"
	@echo "  $(YELLOW)make pipeline$(NC)        - Executa: compile → preprocess → features → train → evaluate"
	@echo "  $(YELLOW)make full-pipeline$(NC)   - Pipeline completo: setup → download → pipeline"
	@echo "  $(YELLOW)make ml-only$(NC)         - Apenas ML: train → evaluate (se já tem features)"
	@echo ""
	@echo "$(BLUE)EXTRAS:$(NC)"
	@echo "  $(YELLOW)make features-all$(NC)    - Gera HOG, LBP e Combined (para ablation)"
	@echo "  $(YELLOW)make ablation$(NC)        - Ablation study (requer features separadas)"
	@echo ""
	@echo "$(BLUE)LIMPEZA:$(NC)"
	@echo "  $(YELLOW)make clean$(NC)           - Remove build/"
	@echo "  $(YELLOW)make clean-models$(NC)    - Remove modelos e resultados"
	@echo "  $(YELLOW)make clean-features$(NC)  - Remove CSVs de features"
	@echo "  $(YELLOW)make clean-all$(NC)       - Remove TUDO (dataset, preprocessed, models)"
	@echo ""
	@echo "$(CYAN)GUIA RÁPIDO:$(NC)"
	@echo "  1. Configure credenciais Kaggle:"
	@echo "     $(YELLOW)export KAGGLE_USERNAME='seu_username'$(NC)"
	@echo "     $(YELLOW)export KAGGLE_KEY='sua_key'$(NC)"
	@echo "  2. Execute: $(YELLOW)make full-pipeline$(NC)"
	@echo ""

setup-system:
	@echo "$(GREEN)=== Verificando OpenCV ===$(NC)"
	@if pkg-config --exists opencv4 2>/dev/null; then \
		echo "$(GREEN)✓ OpenCV já instalado$(NC)"; \
	else \
		echo "$(YELLOW)OpenCV não encontrado, instalando...$(NC)"; \
		if [ ! -f setup.sh ]; then \
			echo "$(RED)✗ Erro: setup.sh não encontrado!$(NC)"; \
			exit 1; \
		fi; \
		chmod +x setup.sh; \
		./setup.sh; \
		echo "$(GREEN)✓ OpenCV instalado$(NC)"; \
	fi

setup:
	@echo "$(GREEN)=== Instalando dependências Python ===$(NC)"
	@python3 -m pip install -q --upgrade pip
	@if [ -f requirements.txt ]; then \
		python3 -m pip install -q -r requirements.txt; \
	else \
		python3 -m pip install -q kaggle python-dotenv numpy pandas scikit-learn opencv-python matplotlib seaborn joblib; \
	fi
	@echo "$(GREEN)✓ Pacotes Python instalados$(NC)"
	@echo ""
	@echo "$(YELLOW)Configure suas credenciais Kaggle:$(NC)"
	@echo "  export KAGGLE_USERNAME='seu_username'"
	@echo "  export KAGGLE_KEY='sua_key'"
	@echo ""
	@echo "Obtenha em: https://www.kaggle.com/settings"

download:
	@if [ -z "$$KAGGLE_USERNAME" ] || [ -z "$$KAGGLE_KEY" ]; then \
		echo "$(RED)✗ Erro: Credenciais Kaggle não configuradas!$(NC)"; \
		echo ""; \
		echo "Execute:"; \
		echo "  export KAGGLE_USERNAME='seu_username'"; \
		echo "  export KAGGLE_KEY='sua_key'"; \
		echo ""; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Baixando dataset do Kaggle ===$(NC)"
	@python3 download_dataset.py
	@echo "$(GREEN)✓ Dataset baixado em: $(DATASET_DIR)/$(NC)"

compile:
	@echo "$(GREEN)=== Compilando código C++ ===$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. > /dev/null 2>&1 && make > /dev/null 2>&1
	@if [ $$? -eq 0 ]; then \
		echo "$(GREEN)✓ Compilação concluída$(NC)"; \
		echo "  - $(BUILD_DIR)/preprocess"; \
		echo "  - $(BUILD_DIR)/feature_extraction"; \
	else \
		echo "$(RED)✗ Erro na compilação!$(NC)"; \
		echo "Execute manualmente: cd build && cmake .. && make"; \
		exit 1; \
	fi

preprocess: compile
	@if [ ! -d "$(DATASET_DIR)/train" ]; then \
		echo "$(RED)✗ Erro: Dataset não encontrado em $(DATASET_DIR)/$(NC)"; \
		echo "Execute: make download"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Pré-processamento de imagens ===$(NC)"
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
	@echo "$(GREEN)✓ Pré-processamento concluído: $(PREPROCESSED_DIR)/$(NC)"

features: compile
	@if [ ! -d "$(DATASET_DIR)" ]; then \
		echo "$(RED)✗ Erro: Dataset não encontrado!$(NC)"; \
		echo "Execute: make download"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Extração de Features (HOG + LBP) ===$(NC)"
	@./$(BUILD_DIR)/feature_extraction
	@if [ -f features_combined.csv ]; then \
		echo "$(GREEN)✓ Features extraídas: features_combined.csv$(NC)"; \
		wc -l features_combined.csv | awk '{print "  Total de amostras: " $$1}'; \
	else \
		echo "$(RED)✗ Erro: features_combined.csv não foi gerado!$(NC)"; \
		exit 1; \
	fi

features-all:
	@echo "$(GREEN)=== Gerando TODAS as features (HOG, LBP, Combined) ===$(NC)"
	@echo ""
	@echo "$(YELLOW)INSTRUÇÕES:$(NC)"
	@echo "1. Edite feature_extraction.cpp"
	@echo "2. Mude a linha: int featureType = X;"
	@echo "   - featureType = 1  →  features_hog.csv"
	@echo "   - featureType = 2  →  features_lbp.csv"
	@echo "   - featureType = 3  →  features_combined.csv"
	@echo "3. Execute: make compile && ./build/feature_extraction"
	@echo "4. Repita para cada tipo"
	@echo ""
	@echo "$(RED)Este é apenas um lembrete. Execute os passos manualmente.$(NC)"

train:
	@if [ ! -f features_combined.csv ]; then \
		echo "$(RED)✗ Erro: features_combined.csv não encontrado!$(NC)"; \
		echo "Execute: make features"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Treinando Classificadores ===$(NC)"
	@python3 train_classifier.py
	@if [ -d "$(MODELS_DIR)" ]; then \
		echo ""; \
		echo "$(GREEN)✓ Modelos salvos em: $(MODELS_DIR)/$(NC)"; \
		ls -lh $(MODELS_DIR)/ | tail -n +2 | awk '{print "  - " $$9 " (" $$5 ")"}'; \
	fi

evaluate:
	@if [ ! -d "$(MODELS_DIR)" ] || [ ! -f "$(MODELS_DIR)/svm_model.pkl" ]; then \
		echo "$(RED)✗ Erro: Modelos não encontrados!$(NC)"; \
		echo "Execute: make train"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Avaliando Modelos ===$(NC)"
	@python3 evaluate_model.py
	@if [ -d "$(RESULTS_DIR)" ]; then \
		echo ""; \
		echo "$(GREEN)✓ Resultados salvos em: $(RESULTS_DIR)/$(NC)"; \
		ls -1 $(RESULTS_DIR)/ | sed 's/^/  - /'; \
	fi

ablation:
	@echo "$(GREEN)=== Ablation Study ===$(NC)"
	@echo ""
	@echo "$(YELLOW)Verificando arquivos necessários...$(NC)"
	@missing=""; \
	for file in features_hog.csv features_lbp.csv features_combined.csv; do \
		if [ ! -f "$$file" ]; then \
			echo "  $(RED)✗ $$file$(NC)"; \
			missing="yes"; \
		else \
			echo "  $(GREEN)✓ $$file$(NC)"; \
		fi; \
	done; \
	echo ""; \
	if [ -n "$$missing" ]; then \
		echo "$(YELLOW)Arquivos faltantes. Execute: make features-all$(NC)"; \
		echo "$(RED)Ablation study será executado com arquivos disponíveis.$(NC)"; \
		echo ""; \
	fi
	@if [ -f train_classifier.py ]; then \
		python3 train_classifier.py; \
		echo "$(GREEN)✓ Ablation study incluído em results.json$(NC)"; \
	else \
		echo "$(RED)✗ train_classifier.py não encontrado!$(NC)"; \
		exit 1; \
	fi

slides:
	@if [ ! -f results.json ]; then \
		echo "$(RED)✗ Erro: results.json não encontrado!$(NC)"; \
		echo "Execute: make train"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Gerando Apresentação ===$(NC)"
	@python3 generate_slides.py
	@if [ -f slides.md ]; then \
		echo ""; \
		echo "$(GREEN)✓ Apresentação gerada: slides.md$(NC)"; \
		echo ""; \
		echo "$(CYAN)Para converter para PDF:$(NC)"; \
		echo "  pandoc slides.md -o apresentacao.pdf -t beamer"; \
	fi

# ========== ATALHOS ==========

pipeline: compile preprocess features train evaluate
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)   ✅ PIPELINE COMPLETO!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(CYAN)Arquivos gerados:$(NC)"
	@echo "  - features_combined.csv"
	@echo "  - models/svm_model.pkl"
	@echo "  - models/rf_model.pkl"
	@echo "  - results.json"
	@echo "  - evaluation_results/*.png"
	@echo ""
	@echo "$(YELLOW)Próximos passos:$(NC)"
	@echo "  1. Ver gráficos: ls $(RESULTS_DIR)/"
	@echo "  2. Gerar slides: make slides"
	@echo "  3. Ablation study: make ablation (opcional)"
	@echo "  4. Preparar vídeo de apresentação"

full-pipeline: setup-system setup download pipeline
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)   ✅ SETUP + PIPELINE COMPLETO!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""

ml-only: train evaluate
	@echo ""
	@echo "$(GREEN)✓ Modelos treinados e avaliados$(NC)"

# ========== LIMPEZA ==========

clean:
	@echo "$(YELLOW)Limpando arquivos de build...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Build limpo$(NC)"

clean-models:
	@echo "$(YELLOW)Removendo modelos e resultados...$(NC)"
	@rm -rf $(MODELS_DIR) $(RESULTS_DIR) results.json slides.md
	@echo "$(GREEN)✓ Modelos e resultados removidos$(NC)"

clean-features:
	@echo "$(YELLOW)Removendo features extraídas...$(NC)"
	@rm -f features_*.csv
	@echo "$(GREEN)✓ Features removidas$(NC)"

clean-preprocessed:
	@echo "$(YELLOW)Removendo imagens pré-processadas...$(NC)"
	@rm -rf $(PREPROCESSED_DIR)
	@echo "$(GREEN)✓ Imagens pré-processadas removidas$(NC)"

clean-all: clean clean-models clean-features clean-preprocessed
	@echo "$(RED)Removendo dataset...$(NC)"
	@rm -rf $(DATASET_DIR)
	@echo "$(GREEN)✓ Tudo removido (projeto resetado)$(NC)"

# ========== STATUS ==========

status:
	@echo "$(GREEN)=== Status do Projeto ===$(NC)"
	@echo ""
	@echo "$(BLUE)Dataset:$(NC)"
	@if [ -d "$(DATASET_DIR)" ]; then \
		echo "  $(GREEN)✓$(NC) $(DATASET_DIR)/"; \
	else \
		echo "  $(RED)✗$(NC) $(DATASET_DIR)/ (execute: make download)"; \
	fi
	@echo ""
	@echo "$(BLUE)Pré-processamento:$(NC)"
	@if [ -d "$(PREPROCESSED_DIR)" ]; then \
		echo "  $(GREEN)✓$(NC) $(PREPROCESSED_DIR)/"; \
	else \
		echo "  $(RED)✗$(NC) $(PREPROCESSED_DIR)/ (execute: make preprocess)"; \
	fi
	@echo ""
	@echo "$(BLUE)Features:$(NC)"
	@if [ -f features_combined.csv ]; then \
		echo "  $(GREEN)✓$(NC) features_combined.csv"; \
	else \
		echo "  $(RED)✗$(NC) features_combined.csv (execute: make features)"; \
	fi
	@if [ -f features_hog.csv ]; then \
		echo "  $(GREEN)✓$(NC) features_hog.csv"; \
	else \
		echo "  $(YELLOW)⊘$(NC) features_hog.csv (opcional)"; \
	fi
	@if [ -f features_lbp.csv ]; then \
		echo "  $(GREEN)✓$(NC) features_lbp.csv"; \
	else \
		echo "  $(YELLOW)⊘$(NC) features_lbp.csv (opcional)"; \
	fi
	@echo ""
	@echo "$(BLUE)Modelos:$(NC)"
	@if [ -d "$(MODELS_DIR)" ]; then \
		echo "  $(GREEN)✓$(NC) $(MODELS_DIR)/"; \
	else \
		echo "  $(RED)✗$(NC) $(MODELS_DIR)/ (execute: make train)"; \
	fi
	@echo ""
	@echo "$(BLUE)Resultados:$(NC)"
	@if [ -d "$(RESULTS_DIR)" ]; then \
		echo "  $(GREEN)✓$(NC) $(RESULTS_DIR)/"; \
	else \
		echo "  $(RED)✗$(NC) $(RESULTS_DIR)/ (execute: make evaluate)"; \
	fi
	@if [ -f results.json ]; then \
		echo "  $(GREEN)✓$(NC) results.json"; \
	else \
		echo "  $(RED)✗$(NC) results.json"; \
	fi
	@echo ""