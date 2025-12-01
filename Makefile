# Makefile - Butterfly Image Classification Pipeline

.PHONY: all setup-system setup download compile preprocess features train evaluate clean help

DATASET_DIR = dataset
PREPROCESSED_DIR = preprocessed
BUILD_DIR = build
MODELS_DIR = models
RESULTS_DIR = evaluation_results

GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
BLUE = \033[0;34m
CYAN = \033[0;36m
NC = \033[0m

all: help

help:
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)  🦋 Butterfly Pipeline$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(BLUE)SETUP INICIAL:$(NC)"
	@echo "  $(YELLOW)make setup-system$(NC)    - Instala OpenCV e OpenMP"
	@echo "  $(YELLOW)make setup$(NC)           - Instala pacotes Python"
	@echo "  $(YELLOW)make download$(NC)        - Baixa dataset do Kaggle"
	@echo ""
	@echo "$(BLUE)PIPELINE RÁPIDO:$(NC)"
	@echo "  $(YELLOW)make compile$(NC)         - Compila C++ (com -O3 e OpenMP)"
	@echo "  $(YELLOW)make features$(NC)        - Extrai features (paralelizado, ~5-8min)"
	@echo "  $(YELLOW)make train$(NC)           - Treina (LinearSVC + LR + RF, ~2-4min)"
	@echo "  $(YELLOW)make evaluate$(NC)        - Avalia modelos"
	@echo ""
	@echo "$(BLUE)ATALHOS:$(NC)"
	@echo "  $(YELLOW)make pipeline$(NC)        - Compile → Features → Train → Evaluate"
	@echo "  $(YELLOW)make full-pipeline$(NC)   - Setup completo + Pipeline"
	@echo "  $(YELLOW)make quick-test$(NC)      - Testa em 100 imagens (rápido)"
	@echo ""
	@echo "$(BLUE)PREDIÇÃO:$(NC)"
	@echo "  $(YELLOW)make predict-one$(NC)     - Testa imagem única (IMAGE=path)"
	@echo "  $(YELLOW)make predict-batch$(NC)   - Prediz em batch paralelo (DIR=path)"
	@echo "  $(YELLOW)make evaluate-prediction(NC) -Testa toda a pasta train"
	@echo ""
	@echo "$(BLUE)LIMPEZA:$(NC)"
	@echo "  $(YELLOW)make clean$(NC)           - Remove build/"
	@echo "  $(YELLOW)make clean-all$(NC)       - Remove tudo (dataset, modelos, features)"
	@echo ""
	@echo "$(CYAN)TEMPO ESTIMADO (Pipeline completo):$(NC)"
	@echo "  • Features: ~5-8 min (vs 40 min antes)"
	@echo "  • Training: ~2-4 min (vs 15-20 min antes)"
	@echo "  • Total: ~8-14 min (vs 60+ min antes)"
	@echo ""

setup-system:
	@echo "$(GREEN)=== Verificando OpenCV e OpenMP ===$(NC)"
	@if pkg-config --exists opencv4 2>/dev/null; then \
		echo "$(GREEN)✓ OpenCV instalado$(NC)"; \
	else \
		echo "$(YELLOW)Instalando OpenCV...$(NC)"; \
		if [ ! -f setup.sh ]; then \
			echo "$(RED)✗ setup.sh não encontrado!$(NC)"; \
			exit 1; \
		fi; \
		chmod +x setup.sh; \
		./setup.sh; \
	fi
	@echo ""
	@echo "$(GREEN)Verificando OpenMP...$(NC)"
	@if echo "#include <omp.h>" | g++ -fopenmp -x c++ - -o /dev/null 2>/dev/null; then \
		echo "$(GREEN)✓ OpenMP disponível (g++)$(NC)"; \
	elif dpkg -l | grep -q libomp-dev; then \
		echo "$(GREEN)✓ libomp-dev instalado$(NC)"; \
	elif [ -f /usr/lib/x86_64-linux-gnu/libomp.so ] || [ -f /usr/lib/libomp.so ]; then \
		echo "$(GREEN)✓ libomp.so encontrado$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  OpenMP não detectado$(NC)"; \
		echo "$(YELLOW)Instalando libomp-dev...$(NC)"; \
		sudo apt-get update -qq && sudo apt-get install -y -qq libomp-dev; \
	fi

setup:
	@echo "$(GREEN)=== Instalando dependências Python ===$(NC)"
	@python3 -m pip install -q --upgrade pip
	@python3 -m pip install -q kaggle python-dotenv numpy pandas scikit-learn opencv-python matplotlib seaborn joblib
	@echo "$(GREEN)✓ Pacotes instalados$(NC)"
	@echo ""
	@echo "$(YELLOW)Configure credenciais Kaggle:$(NC)"
	@echo "  export KAGGLE_USERNAME='seu_username'"
	@echo "  export KAGGLE_KEY='sua_key'"

download:
	@if [ -z "$$KAGGLE_USERNAME" ] || [ -z "$$KAGGLE_KEY" ]; then \
		echo "$(RED)✗ Credenciais Kaggle não configuradas!$(NC)"; \
		echo ""; \
		echo "Execute:"; \
		echo "  export KAGGLE_USERNAME='seu_username'"; \
		echo "  export KAGGLE_KEY='sua_key'"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Baixando dataset ===$(NC)"
	@python3 download_dataset.py
	@echo "$(GREEN)✓ Dataset em: $(DATASET_DIR)/$(NC)"

compile:
	@echo "$(GREEN)=== Compilando C++ (com otimizações) ===$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make
	@if [ $$? -eq 0 ]; then \
		echo "$(GREEN)✓ Compilação concluída$(NC)"; \
		echo "  Flags: -O3 -march=native"; \
		if grep -q "OpenMP encontrado" $(BUILD_DIR)/CMakeCache.txt 2>/dev/null || \
		   strings $(BUILD_DIR)/feature_extraction 2>/dev/null | grep -q "GOMP" 2>/dev/null; then \
			echo "  $(GREEN)✓ OpenMP habilitado (paralelização ativa)$(NC)"; \
		else \
			echo "  $(YELLOW)⚠️  OpenMP não detectado (execução sequencial)$(NC)"; \
		fi; \
	else \
		echo "$(RED)✗ Erro na compilação!$(NC)"; \
		exit 1; \
	fi

preprocess: compile
	@if [ ! -d "$(DATASET_DIR)/train" ]; then \
		echo "$(RED)✗ Dataset não encontrado!$(NC)"; \
		echo "Execute: make download"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Pré-processamento ===$(NC)"
	@mkdir -p $(PREPROCESSED_DIR)
	@echo "$(YELLOW)[1/2] Train...$(NC)"
	@./$(BUILD_DIR)/preprocess $(DATASET_DIR)/train $(PREPROCESSED_DIR)/train
	@echo ""
	@echo "$(YELLOW)[2/2] Test...$(NC)"
	@./$(BUILD_DIR)/preprocess $(DATASET_DIR)/test $(PREPROCESSED_DIR)/test
	@echo ""
	@echo "$(GREEN)✓ Concluído: $(PREPROCESSED_DIR)/$(NC)"

features: compile
	@if [ ! -d "$(DATASET_DIR)" ]; then \
		echo "$(RED)✗ Dataset não encontrado!$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Extração de Features (OTIMIZADA) ===$(NC)"
	@echo "$(CYAN)Features compactas: HOG+LBP+Color (~1200 dims)$(NC)"
	@echo "$(CYAN)Paralelização: OpenMP ativo$(NC)"
	@echo ""
	@./$(BUILD_DIR)/feature_extraction
	@if [ -f features_combined.csv ]; then \
		echo ""; \
		echo "$(GREEN)✓ Features: features_combined.csv$(NC)"; \
		wc -l features_combined.csv | awk '{print "  Amostras: " $1}'; \
	else \
		echo "$(RED)✗ Erro na extração!$(NC)"; \
		exit 1; \
	fi

train:
	@if [ ! -f features_combined.csv ]; then \
		echo "$(RED)✗ features_combined.csv não encontrado!$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Treinamento (OTIMIZADO) ===$(NC)"
	@echo "$(CYAN)Modelos: LinearSVC + LogisticRegression + RandomForest$(NC)"
	@echo ""
	@python3 train_classifier.py
	@if [ -d "$(MODELS_DIR)" ]; then \
		echo ""; \
		echo "$(GREEN)✓ Modelos em: $(MODELS_DIR)/$(NC)"; \
		ls -lh $(MODELS_DIR)/*.pkl 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'; \
	fi

evaluate:
	@if [ ! -d "$(MODELS_DIR)" ]; then \
		echo "$(RED)✗ Modelos não encontrados!$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Avaliação ===$(NC)"
	@if [ -f evaluate_model.py ]; then \
		python3 evaluate_model.py; \
	else \
		echo "$(YELLOW)evaluate_model.py não encontrado, pulando gráficos$(NC)"; \
	fi
	@if [ -f results.json ]; then \
		echo ""; \
		echo "$(GREEN)✓ Resultados:$(NC)"; \
		python3 -c "import json; r=json.load(open('results.json')); \
			[print(f'  {k}: Acc={v[\"accuracy\"]:.4f}, F1={v[\"f1_score\"]:.4f}') \
			for k,v in r.items()]"; \
	fi

# ========== ATALHOS ==========

pipeline: compile features train evaluate
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)   ✅ PIPELINE COMPLETO!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(CYAN)Arquivos gerados:$(NC)"
	@echo "  • features_combined.csv"
	@echo "  • models/best_model.pkl"
	@echo "  • models/svm_model.pkl"
	@echo "  • results.json"

full-pipeline: setup-system setup download pipeline
	@echo ""
	@echo "$(GREEN)✅ SETUP + PIPELINE COMPLETO!$(NC)"

# ========== PREDIÇÃO ==========

predict-one:
	@if [ -z "$(IMAGE)" ]; then \
		echo "$(RED)✗ Especifique IMAGE=path$(NC)"; \
		echo "Exemplo: make predict-one IMAGE=dataset/train/Image_1.jpg"; \
		exit 1; \
	fi
	@python3 predict_butterfly.py --image $(IMAGE)

predict-batch:
	@if [ -z "$(DIR)" ]; then \
		echo "$(RED)✗ Especifique DIR=path$(NC)"; \
		echo "Exemplo: make predict-batch DIR=dataset/test"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Predição em Batch (Paralelo) ===$(NC)"
	@python3 predict_butterfly.py --batch $(DIR) --workers 8

# ========== LIMPEZA ==========

clean:
	@echo "$(YELLOW)Limpando build/$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Build limpo$(NC)"

clean-models:
	@echo "$(YELLOW)Removendo modelos e resultados$(NC)"
	@rm -rf $(MODELS_DIR) $(RESULTS_DIR) results.json
	@echo "$(GREEN)✓ Modelos removidos$(NC)"

clean-features:
	@echo "$(YELLOW)Removendo features$(NC)"
	@rm -f features_*.csv
	@echo "$(GREEN)✓ Features removidas$(NC)"

clean-preprocessed:
	@echo "$(YELLOW)Removendo preprocessed/$(NC)"
	@rm -rf $(PREPROCESSED_DIR)
	@echo "$(GREEN)✓ Preprocessed removido$(NC)"

clean-all: clean clean-models clean-features clean-preprocessed
	@echo "$(RED)Removendo dataset/$(NC)"
	@rm -rf $(DATASET_DIR)
	@echo "$(GREEN)✓ Projeto resetado$(NC)"

# ========== STATUS ==========

status:
	@echo "$(GREEN)=== Status do Projeto ===$(NC)"
	@echo ""
	@echo "$(BLUE)Otimizações:$(NC)"
	@if strings $(BUILD_DIR)/feature_extraction 2>/dev/null | grep -q "GOMP"; then \
		echo "  $(GREEN)✓ OpenMP ativo (paralelização)$(NC)"; \
	else \
		echo "  $(YELLOW)⊘ OpenMP inativo$(NC)"; \
	fi
	@if [ -f $(BUILD_DIR)/CMakeCache.txt ]; then \
		if grep -q "\-O3" $(BUILD_DIR)/CMakeCache.txt; then \
			echo "  $(GREEN)✓ Flags de otimização (-O3)$(NC)"; \
		fi; \
	fi
	@echo ""
	@echo "$(BLUE)Dataset:$(NC)"
	@if [ -d "$(DATASET_DIR)" ]; then \
		count=$$(find $(DATASET_DIR)/train -type f 2>/dev/null | wc -l); \
		echo "  $(GREEN)✓ $(DATASET_DIR)/ ($$count imagens)$(NC)"; \
	else \
		echo "  $(RED)✗ $(DATASET_DIR)/ (execute: make download)$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)Features:$(NC)"
	@if [ -f features_combined.csv ]; then \
		lines=$$(wc -l < features_combined.csv); \
		echo "  $(GREEN)✓ features_combined.csv ($$lines linhas)$(NC)"; \
	else \
		echo "  $(RED)✗ features_combined.csv (execute: make features)$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)Modelos:$(NC)"
	@if [ -d "$(MODELS_DIR)" ]; then \
		echo "  $(GREEN)✓ $(MODELS_DIR)/$(NC)"; \
		ls $(MODELS_DIR)/*.pkl 2>/dev/null | sed 's/^/    /'; \
	else \
		echo "  $(RED)✗ $(MODELS_DIR)/ (execute: make train)$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)Resultados:$(NC)"
	@if [ -f results.json ]; then \
		echo "  $(GREEN)✓ results.json$(NC)"; \
	else \
		echo "  $(RED)✗ results.json$(NC)"; \
	fi

evaluate-prediction:
	@if [ ! -d "$(DATASET_DIR)/train" ]; then \
		echo "$(RED)✗ Dataset não encontrado!$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(DATASET_DIR)/Training_set.csv" ]; then \
		echo "$(RED)✗ Training_set.csv não encontrado!$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Avaliando acurácia do modelo no conjunto de treino ===$(NC)"
	@python3 predict_butterfly.py --evaluate $(DATASET_DIR)/train $(DATASET_DIR)/Training_set.csv