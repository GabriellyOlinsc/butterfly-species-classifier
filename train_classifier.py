#!/usr/bin/env python3
"""
Train Classifier V2 - CORRIGIDO
Normalização adequada + validação rigorosa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import joblib
import json
from pathlib import Path
import time

class ButterflyClassifier:
    def __init__(self, features_file='features_combined.csv'):
        self.features_file = features_file
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()  # NOVO: Normalização de dados
        self.results = {}
        
    def load_data(self):
        """Carrega features do CSV"""
        print("📂 Carregando features...")
        
        df = pd.read_csv(self.features_file, header=None)
        
        # Primeira coluna = label, resto = features
        self.y_raw = df.iloc[:, 0].values
        self.X_raw = df.iloc[:, 1:].values
        
        print(f"  ✓ Shape: {self.X_raw.shape}")
        print(f"  ✓ Features: {self.X_raw.shape[1]}")
        
        # Validar dados
        if np.isnan(self.X_raw).any():
            print("  ⚠️  AVISO: Dados contêm NaN, removendo...")
            valid_mask = ~np.isnan(self.X_raw).any(axis=1)
            self.X_raw = self.X_raw[valid_mask]
            self.y_raw = self.y_raw[valid_mask]
        
        if np.isinf(self.X_raw).any():
            print("  ⚠️  AVISO: Dados contêm Inf, removendo...")
            valid_mask = ~np.isinf(self.X_raw).any(axis=1)
            self.X_raw = self.X_raw[valid_mask]
            self.y_raw = self.y_raw[valid_mask]
        
        # Encode labels
        self.y = self.label_encoder.fit_transform(self.y_raw)
        
        print(f"\n  ✓ Amostras válidas: {len(self.y)}")
        print(f"  ✓ Classes: {len(self.label_encoder.classes_)}")
        
        # Mostrar distribuição
        unique, counts = np.unique(self.y, return_counts=True)
        print(f"\n  Distribuição de classes:")
        for i in range(min(5, len(unique))):
            class_name = self.label_encoder.classes_[unique[i]]
            print(f"    {class_name:30s}: {counts[i]} amostras")
        
        # Validar que todas as classes têm amostras suficientes
        min_samples = counts.min()
        if min_samples < 2:
            print(f"\n  ⚠️  AVISO: Classe com apenas {min_samples} amostra(s)")
            print("  Isso pode causar problemas no split train/test")
        
        return self.X_raw, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split com validação de estratificação
        """
        print(f"\n📊 Separando dados (test_size={test_size})...")
        
        # Verificar se é possível estratificar
        unique, counts = np.unique(self.y, return_counts=True)
        min_count = counts.min()
        
        if min_count < 2:
            print(f"  ⚠️  Classe com {min_count} amostra, estratificação desabilitada")
            stratify = None
        else:
            stratify = self.y
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_raw, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify
        )
        
        print(f"  ✓ Train: {len(X_train)} amostras")
        print(f"  ✓ Test:  {len(X_test)} amostras")
        
        # NORMALIZAÇÃO CRÍTICA: Fit no train, transform em ambos
        print(f"\n  Normalizando features (StandardScaler)...")
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        
        self.y_train = y_train
        self.y_test = y_test
        
        # Estatísticas da normalização
        print(f"    Antes - média: {X_train.mean():.6f}, std: {X_train.std():.6f}")
        print(f"    Depois - média: {self.X_train.mean():.6f}, std: {self.X_train.std():.6f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_svm(self, optimize=False):
        """
        Treina SVM com parâmetros ajustados
        """
        print("\n🤖 Treinando SVM...")
        start = time.time()
        
        if optimize:
            from sklearn.model_selection import GridSearchCV
            print("  (Grid search - pode demorar!)")
            
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf']
            }
            
            svm = GridSearchCV(
                SVC(probability=True, random_state=42, class_weight='balanced'),
                param_grid, 
                cv=3, 
                n_jobs=-1, 
                verbose=1,
                scoring='f1_weighted'
            )
            
            svm.fit(self.X_train, self.y_train)
            print(f"\n  Melhores params: {svm.best_params_}")
            print(f"  Melhor F1 (CV): {svm.best_score_:.4f}")
            
            self.svm = svm.best_estimator_
        else:
            # Parâmetros default melhorados
            self.svm = SVC(
                kernel='rbf',
                C=10.0,           # Regularização moderada
                gamma='scale',    # Adapta ao número de features
                probability=True,
                random_state=42,
                class_weight='balanced'  # CRÍTICO: balancear classes desbalanceadas
            )
            
            self.svm.fit(self.X_train, self.y_train)
        
        elapsed = time.time() - start
        print(f"  ✓ Treinado em {elapsed:.2f}s")
        
        # Cross-validation no train
        print(f"\n  Validação cruzada (3-fold)...")
        cv_scores = cross_val_score(self.svm, self.X_train, self.y_train, 
                                    cv=3, scoring='accuracy', n_jobs=-1)
        print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.svm
    
    def train_random_forest(self, n_estimators=200):
        """Treina Random Forest"""
        print(f"\n🌲 Treinando Random Forest ({n_estimators} árvores)...")
        start = time.time()
        
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=30,           # Aumentado
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # Balancear classes
            verbose=0
        )
        
        self.rf.fit(self.X_train, self.y_train)
        
        elapsed = time.time() - start
        print(f"  ✓ Treinado em {elapsed:.2f}s")
        
        # Cross-validation
        print(f"\n  Validação cruzada (3-fold)...")
        cv_scores = cross_val_score(self.rf, self.X_train, self.y_train, 
                                    cv=3, scoring='accuracy', n_jobs=-1)
        print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.rf
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Avalia modelo no conjunto de teste"""
        print(f"\n📈 Avaliando {model_name} no conjunto de teste...")
        
        # Predições
        y_pred = model.predict(X_test)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        print(f"  ✓ Accuracy:  {acc:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall:    {recall:.4f}")
        print(f"  ✓ F1-Score:  {f1:.4f}")
        
        # Classification report detalhado
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Salvar resultados
        results = {
            'model_name': model_name,
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Análise de erro por classe
        print(f"\n  Classes com pior desempenho:")
        class_f1 = [(cls, report[cls]['f1-score']) 
                    for cls in self.label_encoder.classes_ 
                    if cls in report]
        class_f1.sort(key=lambda x: x[1])
        
        for cls, f1_score in class_f1[:5]:
            print(f"    {cls:30s}: F1 = {f1_score:.4f}")
        
        self.results[model_name] = results
        return results
    
    def save_models(self, output_dir='models'):
        """Salva modelos e scaler"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\n💾 Salvando modelos em '{output_dir}/'...")
        
        if hasattr(self, 'svm'):
            joblib.dump(self.svm, f'{output_dir}/svm_model.pkl')
            print("  ✓ svm_model.pkl")
        
        if hasattr(self, 'rf'):
            joblib.dump(self.rf, f'{output_dir}/rf_model.pkl')
            print("  ✓ rf_model.pkl")
        
        # CRÍTICO: Salvar scaler também!
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        print("  ✓ scaler.pkl")
        
        joblib.dump(self.label_encoder, f'{output_dir}/label_encoder.pkl')
        print("  ✓ label_encoder.pkl")
    
    def save_results(self, output_file='results.json'):
        """Salva métricas"""
        print(f"\n💾 Salvando resultados em '{output_file}'...")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("  ✓ Resultados salvos!")


def main():
    print("=" * 70)
    print("🦋 BUTTERFLY CLASSIFICATION - TRAINING V2")
    print("=" * 70)
    
    # Verificar arquivo de features
    if not Path('features_combined.csv').exists():
        print("\n❌ ERRO: features_combined.csv não encontrado!")
        print("\nExecute primeiro:")
        print("  make compile")
        print("  ./build/feature_extraction")
        return
    
    # Inicializar
    clf = ButterflyClassifier('features_combined.csv')
    
    # 1. Carregar dados
    try:
        X, y = clf.load_data()
    except Exception as e:
        print(f"\n❌ ERRO ao carregar dados: {e}")
        return
    
    # 2. Split
    try:
        clf.split_data(test_size=0.2, random_state=42)
    except Exception as e:
        print(f"\n❌ ERRO no split: {e}")
        return
    
    # 3. Treinar modelos
    print("\n" + "=" * 70)
    print("TREINAMENTO")
    print("=" * 70)
    
    try:
        clf.train_svm(optimize=False)  # Mude para True para otimizar
    except Exception as e:
        print(f"\n❌ ERRO ao treinar SVM: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        clf.train_random_forest(n_estimators=200)
    except Exception as e:
        print(f"\n❌ ERRO ao treinar RF: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Avaliar
    print("\n" + "=" * 70)
    print("AVALIAÇÃO FINAL")
    print("=" * 70)
    
    if hasattr(clf, 'svm'):
        clf.evaluate_model(clf.svm, 'SVM', clf.X_test, clf.y_test)
    
    if hasattr(clf, 'rf'):
        clf.evaluate_model(clf.rf, 'RandomForest', clf.X_test, clf.y_test)
    
    # 5. Salvar
    clf.save_models()
    clf.save_results()
    
    # 6. Resumo
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETO!")
    print("=" * 70)
    print("\nArquivos gerados:")
    print("  • models/svm_model.pkl")
    print("  • models/scaler.pkl (IMPORTANTE!)")
    print("  • models/label_encoder.pkl")
    print("  • results.json")
    print("\nPróximo passo:")
    print("  python3 predict_butterfly.py --image dataset/train/Image_1.jpg")
    print("=" * 70)


if __name__ == '__main__':
    main()