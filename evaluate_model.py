#!/usr/bin/env python3
"""
Evaluate Model - Error Analysis & Visualization
Análise detalhada de erro com visualizações para apresentação
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model_path='models/svm_model.pkl', 
                 encoder_path='models/label_encoder.pkl',
                 features_file='features_combined.csv'):
        
        print("📂 Carregando modelo treinado...")
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Carregar dados de teste
        df = pd.read_csv(features_file, header=None)
        X = df.iloc[:, 1:].values
        y_raw = df.iloc[:, 0].values
        y = self.label_encoder.transform(y_raw)
        
        # Mesmo split do treinamento (seed=42)
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.y_pred = self.model.predict(self.X_test)
        
        print(f"✓ Modelo carregado: {len(self.X_test)} amostras de teste")
        
        # Criar diretório para visualizações
        self.output_dir = Path('evaluation_results')
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_confusion_matrix(self, save=True):
        """Gera heatmap da matriz de confusão"""
        print("\n📊 Gerando matriz de confusão...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(12, 10))
        
        # Normalizar para percentual
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Percentual (%)'})
        
        plt.title('Matriz de Confusão - Classificação de Borboletas\n(% por linha)', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Classe Real', fontsize=12)
        plt.xlabel('Classe Predita', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Salvo: {output_path}")
        
        plt.close()
    
    def plot_per_class_metrics(self, save=True):
        """Gráfico de barras com Precision/Recall/F1 por classe"""
        print("\n📊 Gerando métricas por classe...")
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        classes = self.label_encoder.classes_
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
        ax.bar(x, recall, width, label='Recall', color='#3498db')
        ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Espécie de Borboleta', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Métricas de Classificação por Espécie', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'per_class_metrics.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Salvo: {output_path}")
        
        plt.close()
    
    def analyze_errors(self):
        """Identifica padrões de erro"""
        print("\n🔍 Analisando padrões de erro...")
        
        errors = []
        corrects = []
        
        for i, (true_label, pred_label) in enumerate(zip(self.y_test, self.y_pred)):
            if true_label != pred_label:
                errors.append({
                    'index': i,
                    'true': self.label_encoder.classes_[true_label],
                    'predicted': self.label_encoder.classes_[pred_label]
                })
            else:
                corrects.append({
                    'index': i,
                    'class': self.label_encoder.classes_[true_label]
                })
        
        print(f"\n📈 Estatísticas de Erro:")
        print(f"  Total de amostras: {len(self.y_test)}")
        print(f"  Acertos: {len(corrects)} ({len(corrects)/len(self.y_test)*100:.2f}%)")
        print(f"  Erros: {len(errors)} ({len(errors)/len(self.y_test)*100:.2f}%)")
        
        # Confusões mais comuns
        if errors:
            error_pairs = {}
            for e in errors:
                key = f"{e['true']} → {e['predicted']}"
                error_pairs[key] = error_pairs.get(key, 0) + 1
            
            sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n❌ Top 5 Confusões Mais Comuns:")
            for pair, count in sorted_errors[:5]:
                print(f"  {pair}: {count} vezes")
        
        # Salvar lista completa de erros
        error_report_path = self.output_dir / 'error_analysis.txt'
        with open(error_report_path, 'w') as f:
            f.write("ANÁLISE DETALHADA DE ERROS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total de erros: {len(errors)}\n")
            f.write(f"Acurácia: {len(corrects)/len(self.y_test)*100:.2f}%\n\n")
            
            f.write("CONFUSÕES MAIS FREQUENTES:\n")
            f.write("-" * 60 + "\n")
            for pair, count in sorted_errors[:10]:
                f.write(f"{pair}: {count} vezes\n")
            
            f.write("\n\nLISTA COMPLETA DE ERROS:\n")
            f.write("-" * 60 + "\n")
            for e in errors:
                f.write(f"Amostra {e['index']}: {e['true']} classificado como {e['predicted']}\n")
        
        print(f"\n  ✓ Relatório salvo: {error_report_path}")
        
        return errors, corrects
    
    def plot_class_distribution(self, save=True):
        """Distribuição de amostras por classe"""
        print("\n📊 Gerando distribuição de classes...")
        
        classes, counts = np.unique(self.y_test, return_counts=True)
        class_names = [self.label_encoder.classes_[c] for c in classes]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, counts, color='steelblue', edgecolor='navy')
        
        # Adicionar valores no topo das barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Espécie', fontsize=12)
        plt.ylabel('Número de Amostras', fontsize=12)
        plt.title('Distribuição de Classes no Conjunto de Teste', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'class_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Salvo: {output_path}")
        
        plt.close()
    
    def compare_ablation_results(self, results_file='results.json', save=True):
        """Visualiza resultados do ablation study"""
        print("\n📊 Comparando ablation study...")
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            if 'ablation_study' not in results:
                print("  ⚠️  Ablation study não encontrado em results.json")
                return
            
            ablation = results['ablation_study']
            
            # Preparar dados
            methods = []
            f1_scores = []
            colors = []
            
            if 'baseline' in ablation:
                methods.append('HOG + LBP\n(Baseline)')
                f1_scores.append(ablation['baseline'])
                colors.append('#2ecc71')
            
            if 'only_hog' in ablation:
                methods.append('Apenas\nHOG')
                f1_scores.append(ablation['only_hog'])
                colors.append('#e67e22')
            
            if 'only_lbp' in ablation:
                methods.append('Apenas\nLBP')
                f1_scores.append(ablation['only_lbp'])
                colors.append('#3498db')
            
            # Plot
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
            
            # Valores no topo
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.ylabel('F1-Score', fontsize=12)
            plt.title('Ablation Study - Impacto de Cada Componente\n(Quanto maior, melhor)', 
                     fontsize=14, fontweight='bold')
            plt.ylim(0, max(f1_scores) * 1.15)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            if save:
                output_path = self.output_dir / 'ablation_study.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"  ✓ Salvo: {output_path}")
            
            plt.close()
            
        except FileNotFoundError:
            print(f"  ⚠️  Arquivo {results_file} não encontrado")
    
    def generate_summary_report(self, results_file='results.json'):
        """Gera relatório textual completo"""
        print("\n📄 Gerando relatório final...")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RELATÓRIO FINAL - CLASSIFICAÇÃO DE BORBOLETAS\n")
            f.write("=" * 70 + "\n\n")
            
            # Resultados dos modelos
            for model_name, metrics in results.items():
                if model_name == 'ablation_study':
                    continue
                
                f.write(f"\n{model_name.upper()}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall:    {metrics['recall']:.4f}\n")
                f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
                if 'auc' in metrics:
                    f.write(f"AUC:       {metrics['auc']:.4f}\n")
            
            # Ablation Study
            if 'ablation_study' in results:
                f.write("\n\nABLATION STUDY\n")
                f.write("-" * 70 + "\n")
                for method, f1 in results['ablation_study'].items():
                    f.write(f"{method:20s}: F1 = {f1:.4f}\n")
            
            # Métricas por classe
            if 'SVM' in results:
                report_dict = results['SVM']['classification_report']
                f.write("\n\nMÉTRICAS POR CLASSE (SVM)\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Classe':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
                f.write("-" * 70 + "\n")
                
                for class_name in self.label_encoder.classes_:
                    if class_name in report_dict:
                        p = report_dict[class_name]['precision']
                        r = report_dict[class_name]['recall']
                        f1 = report_dict[class_name]['f1-score']
                        f.write(f"{class_name:<30} {p:>10.4f} {r:>10.4f} {f1:>10.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Visualizações salvas em: evaluation_results/\n")
            f.write("=" * 70 + "\n")
        
        print(f"  ✓ Relatório salvo: {report_path}")


def main():
    print("=" * 60)
    print("🦋 ANÁLISE DE ERRO E VISUALIZAÇÕES")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    
    # Gerar todas as visualizações
    evaluator.plot_confusion_matrix()
    evaluator.plot_per_class_metrics()
    evaluator.plot_class_distribution()
    evaluator.compare_ablation_results()
    
    # Análise de erro
    evaluator.analyze_errors()
    
    # Relatório final
    evaluator.generate_summary_report()
    
    print("\n" + "=" * 60)
    print("✅ ANÁLISE COMPLETA!")
    print("=" * 60)
    print("\nTodos os resultados estão em: evaluation_results/")
    print("\nArquivos gerados:")
    print("  • confusion_matrix.png")
    print("  • per_class_metrics.png")
    print("  • class_distribution.png")
    print("  • ablation_study.png")
    print("  • error_analysis.txt")
    print("  • summary_report.txt")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()