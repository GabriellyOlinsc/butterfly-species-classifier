#!/usr/bin/env python3
"""
Train Classifier - Butterfly Species Classification
Treina modelos clássicos (SVM, RF) comparando resultados
Versão CORRIGIDA - Ablation study e pós-processamento
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import joblib
import json
from pathlib import Path
import time

class ButterflyClassifier:
    def __init__(self, features_file='features_combined.csv'):
        """
        Inicializa o classificador de borboletas.
        
        Args:
            features_file: Arquivo CSV com features extraídas (label, feature1, feature2, ...)
        """
        self.features_file = features_file
        self.label_encoder = LabelEncoder()
        self.results = {}
        
    def load_data(self):
        """
        Carrega features do CSV gerado pelo C++.
        
        Formato esperado:
        - Primeira coluna: label (nome da espécie)
        - Demais colunas: features numéricas (HOG + LBP)
        
        Returns:
            X: array de features
            y: array de labels (encoded)
        """
        print("📂 Carregando features...")
        df = pd.read_csv(self.features_file, header=None)
        
        # Primeira coluna = label, resto = features
        self.y_raw = df.iloc[:, 0].values
        self.X = df.iloc[:, 1:].values
        
        # Encode labels para números
        self.y = self.label_encoder.fit_transform(self.y_raw)
        
        print(f"✓ Dataset carregado: {self.X.shape[0]} imagens, {self.X.shape[1]} features")
        print(f"✓ Classes encontradas: {len(self.label_encoder.classes_)}")
        print(f"  Espécies: {', '.join(self.label_encoder.classes_[:5])}...")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Separa train/val/test com semente fixa (reprodutibilidade).
        
        Args:
            test_size: Proporção do conjunto de teste
            val_size: Proporção do conjunto de validação
            random_state: Semente para reprodutibilidade
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\n📊 Separando dataset...")
        
        # Train + temp (que vira val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=(test_size + val_size), 
            random_state=random_state, stratify=self.y
        )
        
        # Val + Test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"✓ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_svm(self, use_grid_search=False):
        """
        Treina SVM (baseline recomendado).
        
        Por que SVM?
        - Eficaz com features de alta dimensão (HOG + LBP)
        - Kernel RBF captura relações não-lineares
        - Robusto a overfitting com regularização C
        
        Args:
            use_grid_search: Se True, otimiza hiperparâmetros (demorado)
        
        Returns:
            Modelo SVM treinado
        """
        print("\n🤖 Treinando SVM...")
        start = time.time()
        
        if use_grid_search:
            print("  (Grid search ativado - pode demorar!)")
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
            svm = GridSearchCV(SVC(probability=True, random_state=42), 
                             param_grid, cv=3, n_jobs=-1, verbose=1)
        else:
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', 
                     probability=True, random_state=42)
        
        svm.fit(self.X_train, self.y_train)
        
        if use_grid_search:
            print(f"  Melhores params: {svm.best_params_}")
            self.svm = svm.best_estimator_
        else:
            self.svm = svm
        
        elapsed = time.time() - start
        print(f"✓ SVM treinado em {elapsed:.2f}s")
        
        return self.svm
    
    def train_random_forest(self, n_estimators=100):
        """
        Treina Random Forest (alternativa ao SVM).
        
        Por que Random Forest?
        - Ensemble de árvores (robustez)
        - Não precisa de normalização de features
        - Feature importance automática
        
        Args:
            n_estimators: Número de árvores na floresta
        
        Returns:
            Modelo Random Forest treinado
        """
        print(f"\n🌲 Treinando Random Forest ({n_estimators} árvores)...")
        start = time.time()
        
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.rf.fit(self.X_train, self.y_train)
        
        elapsed = time.time() - start
        print(f"✓ Random Forest treinado em {elapsed:.2f}s")
        
        return self.rf
    
    def post_process_predictions(self, y_pred, y_proba, threshold=0.6):
        """
        Pós-processamento: Identifica predições com baixa confiança.
        
        Por que pós-processar?
        - Detectar casos ambíguos (ex: probabilidade 0.51 vs 0.49)
        - Permite revisão manual ou re-classificação
        - Melhora confiabilidade do sistema
        
        Args:
            y_pred: Predições do modelo
            y_proba: Probabilidades de cada classe
            threshold: Confiança mínima aceitável
        
        Returns:
            y_pred: Predições (mantidas)
            uncertain_mask: Array booleano indicando predições incertas
            confidence: Confiança de cada predição
        """
        confidence = np.max(y_proba, axis=1)
        uncertain_mask = confidence < threshold
        
        n_uncertain = np.sum(uncertain_mask)
        print(f"\n🔍 Pós-processamento:")
        print(f"  Predições incertas (< {threshold}): {n_uncertain}/{len(y_pred)} ({n_uncertain/len(y_pred)*100:.2f}%)")
        
        return y_pred, uncertain_mask, confidence
    
    def evaluate_model(self, model, model_name, X_test, y_test, apply_postprocessing=True):
        """
        Avalia modelo e retorna todas as métricas.
        
        Métricas incluídas:
        - Accuracy: Taxa de acerto geral
        - Precision: Dentre as predições positivas, quantas corretas
        - Recall: Dentre os verdadeiros positivos, quantos detectados
        - F1-Score: Média harmônica de precision e recall
        - AUC: Área sob a curva ROC (classificação multi-classe)
        
        Args:
            model: Modelo treinado
            model_name: Nome do modelo (para logging)
            X_test: Features de teste
            y_test: Labels verdadeiros
            apply_postprocessing: Se True, aplica pós-processamento
        
        Returns:
            Dicionário com todas as métricas
        """
        print(f"\n📈 Avaliando {model_name}...")
        
        # Predições
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Pós-processamento (NOVO!)
        uncertain_mask = None
        confidence = None
        if apply_postprocessing and y_proba is not None:
            y_pred, uncertain_mask, confidence = self.post_process_predictions(
                y_pred, y_proba, threshold=0.6
            )
        
        # Métricas básicas
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification Report (por classe)
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        results = {
            'model_name': model_name,
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Adicionar informações de pós-processamento
        if uncertain_mask is not None:
            results['postprocessing'] = {
                'n_uncertain': int(np.sum(uncertain_mask)),
                'percent_uncertain': float(np.sum(uncertain_mask) / len(y_pred) * 100),
                'avg_confidence': float(np.mean(confidence))
            }
        
        # AUC (se tiver probabilidades)
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                results['auc'] = float(auc)
            except:
                pass
        
        print(f"✓ Accuracy: {acc:.4f}")
        print(f"✓ Precision: {precision:.4f}")
        print(f"✓ Recall: {recall:.4f}")
        print(f"✓ F1-Score: {f1:.4f}")
        
        self.results[model_name] = results
        return results
    
    def ablation_study(self):
        """
        Estudo de ablação - testa pipeline sem componentes.
        
        CORRIGIDO: Usa train_test_split corretamente para cada arquivo.
        
        Objetivo:
        - Medir impacto de cada componente (HOG vs LBP)
        - Justificar decisão de combinar features
        - Demonstrar que HOG + LBP > HOG isolado ou LBP isolado
        
        Returns:
            Dicionário com F1-scores de cada configuração
        """
        print("\n🔬 ABLATION STUDY")
        print("=" * 50)
        
        ablation_results = {}
        
        # 1. Baseline completo (HOG + LBP)
        print("\n[1/3] Baseline: HOG + LBP combinados")
        if hasattr(self, 'svm'):
            results = self.evaluate_model(self.svm, 'SVM_Baseline', 
                                        self.X_test, self.y_test,
                                        apply_postprocessing=False)
            ablation_results['baseline'] = results['f1_score']
        
        # 2. Apenas HOG (CORRIGIDO!)
        print("\n[2/3] Ablation: Apenas HOG (sem LBP)")
        try:
            df_hog = pd.read_csv('features_hog.csv', header=None)
            X_hog = df_hog.iloc[:, 1:].values
            y_hog_raw = df_hog.iloc[:, 0].values
            y_hog = self.label_encoder.transform(y_hog_raw)
            
            # CORREÇÃO: Split correto com train_test_split
            X_hog_train, X_hog_test, y_hog_train, y_hog_test = train_test_split(
                X_hog, y_hog, test_size=0.2, random_state=42, stratify=y_hog
            )
            
            # Treinar SVM apenas com HOG
            svm_hog = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            svm_hog.fit(X_hog_train, y_hog_train)
            
            # Avaliar
            y_pred_hog = svm_hog.predict(X_hog_test)
            _, _, f1_hog, _ = precision_recall_fscore_support(
                y_hog_test, y_pred_hog, average='weighted'
            )
            ablation_results['only_hog'] = float(f1_hog)
            print(f"  ✓ F1-Score (HOG): {f1_hog:.4f}")
            
        except FileNotFoundError:
            print("  ⚠️  features_hog.csv não encontrado")
            print("     Execute: Mude featureType=1 em feature_extraction.cpp e compile")
        except Exception as e:
            print(f"  ❌ Erro ao processar HOG: {e}")
        
        # 3. Apenas LBP (CORRIGIDO!)
        print("\n[3/3] Ablation: Apenas LBP (sem HOG)")
        try:
            df_lbp = pd.read_csv('features_lbp.csv', header=None)
            X_lbp = df_lbp.iloc[:, 1:].values
            y_lbp_raw = df_lbp.iloc[:, 0].values
            y_lbp = self.label_encoder.transform(y_lbp_raw)
            
            X_lbp_train, X_lbp_test, y_lbp_train, y_lbp_test = train_test_split(
                X_lbp, y_lbp, test_size=0.2, random_state=42, stratify=y_lbp
            )
            
            # Treinar SVM apenas com LBP
            svm_lbp = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            svm_lbp.fit(X_lbp_train, y_lbp_train)
            
            # Avaliar
            y_pred_lbp = svm_lbp.predict(X_lbp_test)
            _, _, f1_lbp, _ = precision_recall_fscore_support(
                y_lbp_test, y_pred_lbp, average='weighted'
            )
            ablation_results['only_lbp'] = float(f1_lbp)
            print(f"  ✓ F1-Score (LBP): {f1_lbp:.4f}")
            
        except FileNotFoundError:
            print("  ⚠️  features_lbp.csv não encontrado")
            print("     Execute: Mude featureType=2 em feature_extraction.cpp e compile")
        except Exception as e:
            print(f"  ❌ Erro ao processar LBP: {e}")
        
        self.results['ablation_study'] = ablation_results
        
        # Resumo comparativo
        print("\n" + "=" * 50)
        print("RESUMO ABLATION STUDY:")
        print("=" * 50)
        
        if len(ablation_results) > 0:
            sorted_results = sorted(ablation_results.items(), key=lambda x: x[1], reverse=True)
            for i, (key, value) in enumerate(sorted_results):
                symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"{symbol} {key:20s}: F1 = {value:.4f}")
            
            # Análise de ganho
            if 'baseline' in ablation_results:
                baseline_f1 = ablation_results['baseline']
                
                if 'only_hog' in ablation_results:
                    gain_over_hog = baseline_f1 - ablation_results['only_hog']
                    print(f"\n📊 Ganho de HOG+LBP sobre HOG: +{gain_over_hog:.4f} ({gain_over_hog/ablation_results['only_hog']*100:.2f}%)")
                
                if 'only_lbp' in ablation_results:
                    gain_over_lbp = baseline_f1 - ablation_results['only_lbp']
                    print(f"📊 Ganho de HOG+LBP sobre LBP: +{gain_over_lbp:.4f} ({gain_over_lbp/ablation_results['only_lbp']*100:.2f}%)")
        else:
            print("⚠️  Nenhum resultado de ablação disponível")
            print("   Certifique-se de ter features_hog.csv e features_lbp.csv")
        
        print("=" * 50)
        
        return ablation_results
    
    def save_models(self, output_dir='models'):
        """
        Salva modelos treinados para uso posterior.
        
        Args:
            output_dir: Diretório para salvar os modelos
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\n💾 Salvando modelos em '{output_dir}/'...")
        
        if hasattr(self, 'svm'):
            joblib.dump(self.svm, f'{output_dir}/svm_model.pkl')
            print("  ✓ svm_model.pkl")
        
        if hasattr(self, 'rf'):
            joblib.dump(self.rf, f'{output_dir}/rf_model.pkl')
            print("  ✓ rf_model.pkl")
        
        joblib.dump(self.label_encoder, f'{output_dir}/label_encoder.pkl')
        print("  ✓ label_encoder.pkl")
    
    def save_results(self, output_file='results.json'):
        """
        Salva todas as métricas em JSON para análise posterior.
        
        Args:
            output_file: Nome do arquivo JSON
        """
        print(f"\n💾 Salvando resultados em '{output_file}'...")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("  ✓ Métricas salvas!")


def main():
    """
    Pipeline principal de treinamento.
    
    Etapas:
    1. Carregar features extraídas (HOG + LBP)
    2. Separar train/val/test
    3. Treinar SVM e Random Forest
    4. Avaliar no conjunto de teste
    5. Ablation study (comparar HOG vs LBP vs HOG+LBP)
    6. Salvar modelos e resultados
    """
    print("=" * 60)
    print("🦋 BUTTERFLY SPECIES CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Inicializar
    clf = ButterflyClassifier('features_combined.csv')
    
    # 1. Carregar dados
    X, y = clf.load_data()
    
    # 2. Split train/val/test
    clf.split_data(test_size=0.2, val_size=0.1)
    
    # 3. Treinar modelos
    clf.train_svm(use_grid_search=False)  # Mude para True se quiser otimizar
    clf.train_random_forest(n_estimators=100)
    
    # 4. Avaliar no conjunto de teste
    print("\n" + "=" * 60)
    print("📊 AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print("=" * 60)
    
    clf.evaluate_model(clf.svm, 'SVM', clf.X_test, clf.y_test, apply_postprocessing=True)
    clf.evaluate_model(clf.rf, 'RandomForest', clf.X_test, clf.y_test, apply_postprocessing=True)
    
    # 5. Ablation Study
    clf.ablation_study()
    
    # 6. Salvar tudo
    clf.save_models()
    clf.save_results()
    
    # 7. Resumo final
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETO!")
    print("=" * 60)
    print("\nModelos salvos em: models/")
    print("Resultados salvos em: results.json")
    print("\nPróximo passo: python evaluate_model.py")
    print("=" * 60)


if __name__ == '__main__':
    main()