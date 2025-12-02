#!/usr/bin/env python3
"""
Evaluate Model - Error Analysis & Visualization
Análise detalhada de erro com visualizações para apresentação
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model_path='models/svm_model.pkl', 
                 encoder_path='models/label_encoder.pkl',
                 scaler_path='models/scaler.pkl',
                 features_file='features_combined.csv'):
        
        print("📂 Carregando modelo treinado...")
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.scaler = joblib.load(scaler_path)
        
        # Carregar dados de teste
        df = pd.read_csv(features_file, header=None)
        X = df.iloc[:, 1:].values
        y_raw = df.iloc[:, 0].values
        y = self.label_encoder.transform(y_raw)
        
        # Mesmo split do treinamento (seed=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Aplicar mesmo scaler
        self.X_test = self.scaler.transform(X_test)
        self.y_test = y_test
    
        self.y_pred = self.model.predict(self.X_test)
        
        print(f"✓ Modelo carregado: {len(self.X_test)} amostras de teste")
        print(f"✓ Classes: {len(self.label_encoder.classes_)}")
        
        # Criar diretório para visualizações
        self.output_dir = Path('evaluation_results')
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_confusion_matrix_aggregated(self, save=True):
        """Matriz de confusão SIMPLIFICADA - apenas diagonal vs resto"""
        print("\n📊 Gerando matriz de confusão...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Métricas agregadas
        correct_predictions = np.diag(cm).sum()
        total_predictions = cm.sum()
        
        # Criar matriz 2x2 simplificada
        diagonal = np.diag(cm).sum()
        off_diagonal = total_predictions - diagonal
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Matriz simplificada 2x2
        simple_cm = np.array([[diagonal, 0], [off_diagonal, 0]])
        labels = ['Correto', 'Incorreto']
        
        im1 = ax1.imshow(simple_cm, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(labels)
        ax1.set_yticklabels(['Predições', ''])
        
        # Adicionar números
        for i in range(2):
            for j in range(2):
                if simple_cm[i, j] > 0:
                    text = ax1.text(j, i, f'{int(simple_cm[i, j])}\n({simple_cm[i, j]/total_predictions*100:.1f}%)',
                                   ha="center", va="center", color="black", fontsize=14, fontweight='bold')
        
        ax1.set_title(f'Visão Geral de Acertos\nAccuracy: {correct_predictions/total_predictions*100:.1f}%', 
                     fontsize=12, fontweight='bold')
        
        # Plot 2: Distribuição de erros por classe (Top 15 piores)
        class_errors = []
        for i in range(len(self.label_encoder.classes_)):
            total_class = cm[i, :].sum()
            errors_class = total_class - cm[i, i]
            if total_class > 0:
                error_rate = errors_class / total_class
                class_errors.append((self.label_encoder.classes_[i], error_rate, errors_class, total_class))
        
        # Ordenar por taxa de erro
        class_errors.sort(key=lambda x: x[1], reverse=True)
        top_errors = class_errors[:15]
        
        classes = [x[0][:25] for x in top_errors]  # Truncar nomes longos
        error_rates = [x[1] * 100 for x in top_errors]
        
        colors = ['#e74c3c' if er > 70 else '#f39c12' if er > 50 else '#3498db' for er in error_rates]
        
        bars = ax2.barh(classes, error_rates, color=colors, edgecolor='black')
        ax2.set_xlabel('Taxa de Erro (%)', fontsize=11)
        ax2.set_title('Top 15 Classes com Maior Taxa de Erro', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (bar, (cls, rate, errs, tot)) in enumerate(zip(bars, top_errors)):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{rate*100:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'confusion_matrix_simplified.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Salvo: {output_path}")
        
        plt.close()
    
    def plot_top_bottom_classes(self, save=True):
        """Mostra apenas as melhores e piores classes"""
        print("\n📊 Gerando comparação top/bottom classes...")
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Coletar F1-scores
        class_metrics = []
        for cls in self.label_encoder.classes_:
            if cls in report:
                f1 = report[cls]['f1-score']
                precision = report[cls]['precision']
                recall = report[cls]['recall']
                support = report[cls]['support']
                class_metrics.append((cls, f1, precision, recall, support))
        
        # Ordenar por F1
        class_metrics.sort(key=lambda x: x[1])
        
        # Top 10 melhores e 10 piores
        bottom_10 = class_metrics[:10]
        top_10 = class_metrics[-10:]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Piores 10
        classes_bottom = [x[0][:20] for x in bottom_10]
        f1_bottom = [x[1] for x in bottom_10]
        precision_bottom = [x[2] for x in bottom_10]
        recall_bottom = [x[3] for x in bottom_10]
        
        x_bottom = np.arange(len(classes_bottom))
        width = 0.25
        
        ax1.barh(x_bottom - width, precision_bottom, width, label='Precision', color='#e74c3c', alpha=0.8)
        ax1.barh(x_bottom, recall_bottom, width, label='Recall', color='#f39c12', alpha=0.8)
        ax1.barh(x_bottom + width, f1_bottom, width, label='F1-Score', color='#c0392b', alpha=0.8)
        
        ax1.set_yticks(x_bottom)
        ax1.set_yticklabels(classes_bottom, fontsize=9)
        ax1.set_xlabel('Score', fontsize=11)
        ax1.set_title('❌ 10 Piores Classes (menor F1-Score)', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlim(0, 1)
        
        # Plot 2: Melhores 10
        classes_top = [x[0][:20] for x in top_10]
        f1_top = [x[1] for x in top_10]
        precision_top = [x[2] for x in top_10]
        recall_top = [x[3] for x in top_10]
        
        x_top = np.arange(len(classes_top))
        
        ax2.barh(x_top - width, precision_top, width, label='Precision', color='#2ecc71', alpha=0.8)
        ax2.barh(x_top, recall_top, width, label='Recall', color='#3498db', alpha=0.8)
        ax2.barh(x_top + width, f1_top, width, label='F1-Score', color='#27ae60', alpha=0.8)
        
        ax2.set_yticks(x_top)
        ax2.set_yticklabels(classes_top, fontsize=9)
        ax2.set_xlabel('Score', fontsize=11)
        ax2.set_title('✅ 10 Melhores Classes (maior F1-Score)', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'top_bottom_classes.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Salvo: {output_path}")
        
        plt.close()
    
    def plot_performance_summary(self, save=True):
        """Dashboard resumido com métricas principais"""
        print("\n📊 Gerando dashboard de performance...")
        
        # Calcular métricas
        accuracy = accuracy_score(self.y_test, self.y_pred)
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Distribuição de F1-scores
        f1_scores = [report[cls]['f1-score'] for cls in self.label_encoder.classes_ if cls in report]
        
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Métricas gerais (texto grande)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        metrics_text = f"""
        MÉTRICAS GERAIS
        ═══════════════════════════
        
        Accuracy:     {accuracy:.2%}
        Precision:    {report['weighted avg']['precision']:.2%}
        Recall:       {report['weighted avg']['recall']:.2%}
        F1-Score:     {report['weighted avg']['f1-score']:.2%}
        
        Classes:      {len(self.label_encoder.classes_)}
        Amostras:     {len(self.y_test)}
        """
        
        ax1.text(0.1, 0.5, metrics_text, fontsize=14, family='monospace',
                verticalalignment='center', fontweight='bold')
        
        # Plot 2: Distribuição de F1-scores
        ax2 = fig.add_subplot(gs[0, 1])
        
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, edges = np.histogram(f1_scores, bins=bins)
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
        
        bars = ax2.bar(range(len(hist)), hist, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(hist)))
        ax2.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
        ax2.set_xlabel('F1-Score', fontsize=11)
        ax2.set_ylabel('Número de Classes', fontsize=11)
        ax2.set_title('Distribuição de F1-Scores por Classe', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Top 5 confusões
        ax3 = fig.add_subplot(gs[1, :])
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Encontrar maiores confusões (fora da diagonal)
        confusions = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    true_class = self.label_encoder.classes_[i]
                    pred_class = self.label_encoder.classes_[j]
                    confusions.append((true_class, pred_class, cm[i, j]))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        top_5 = confusions[:10]
        
        if top_5:
            labels = [f"{t[:15]} → {p[:15]}" for t, p, _ in top_5]
            counts = [c for _, _, c in top_5]
            
            bars = ax3.barh(labels, counts, color='#e74c3c', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Número de Confusões', fontsize=11)
            ax3.set_title('Top 10 Confusões Mais Frequentes', fontsize=12, fontweight='bold')
            ax3.invert_yaxis()
            ax3.grid(axis='x', alpha=0.3)
            
            # Adicionar valores
            for bar in bars:
                width = bar.get_width()
                ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}',
                        ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.suptitle('Dashboard de Performance do Modelo', fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            output_path = self.output_dir / 'performance_dashboard.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Salvo: {output_path}")
        
        plt.close()
    
    def analyze_errors(self):
        """Identifica padrões de erro - VERSÃO CORRIGIDA"""
        print("\n🔍 Analisando padrões de erro...")
        
        # Contar acertos e erros
        correct_mask = (self.y_test == self.y_pred)
        n_correct = correct_mask.sum()
        n_errors = len(self.y_test) - n_correct
        
        accuracy = n_correct / len(self.y_test)
        
        print(f"\n📈 Estatísticas de Erro (Conjunto de Teste = 20% dos dados):")
        print(f"  Total de amostras de teste: {len(self.y_test)}")
        print(f"  Acertos: {n_correct} ({accuracy*100:.2f}%)")
        print(f"  Erros: {n_errors} ({(1-accuracy)*100:.2f}%)")
        
        # Analisar confusões mais comuns
        error_pairs = {}
        for i in range(len(self.y_test)):
            if self.y_test[i] != self.y_pred[i]:
                true_label = self.label_encoder.classes_[self.y_test[i]]
                pred_label = self.label_encoder.classes_[self.y_pred[i]]
                key = f"{true_label} → {pred_label}"
                error_pairs[key] = error_pairs.get(key, 0) + 1
        
        if error_pairs:
            sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n❌ Top 10 Confusões Mais Comuns:")
            for i, (pair, count) in enumerate(sorted_errors[:10], 1):
                print(f"  {i:2d}. {pair}: {count} vezes ({count/n_errors*100:.1f}% dos erros)")
        
        # Salvar lista completa de erros
        error_report_path = self.output_dir / 'error_analysis.txt'
        with open(error_report_path, 'w') as f:
            f.write("ANÁLISE DETALHADA DE ERROS (Conjunto de Teste)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total de amostras de teste: {len(self.y_test)}\n")
            f.write(f"Acertos: {n_correct} ({accuracy*100:.2f}%)\n")
            f.write(f"Erros: {n_errors} ({(1-accuracy)*100:.2f}%)\n\n")
            
            f.write("CONFUSÕES MAIS FREQUENTES:\n")
            f.write("-" * 80 + "\n")
            for i, (pair, count) in enumerate(sorted_errors[:30], 1):
                f.write(f"{i:2d}. {pair}: {count} vezes ({count/n_errors*100:.1f}% dos erros)\n")
            
            f.write("\n\nDISTRIBUIÇÃO DE ERROS POR CLASSE:\n")
            f.write("-" * 80 + "\n")
            
            # Erros por classe verdadeira
            error_counts = {}
            total_counts = {}
            for i in range(len(self.y_test)):
                true_label = self.label_encoder.classes_[self.y_test[i]]
                total_counts[true_label] = total_counts.get(true_label, 0) + 1
                if self.y_test[i] != self.y_pred[i]:
                    error_counts[true_label] = error_counts.get(true_label, 0) + 1
            
            sorted_class_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            for cls, count in sorted_class_errors:
                total = total_counts.get(cls, 1)
                error_rate = count / total * 100
                f.write(f"{cls:30s}: {count}/{total} erros ({error_rate:.1f}%)\n")
        
        print(f"\n  ✓ Relatório detalhado salvo: {error_report_path}")
        
        return n_correct, n_errors
    
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
            
            f.write("IMPORTANTE: Todos os resultados são baseados no conjunto de TESTE\n")
            f.write("(20% dos dados, nunca vistos durante o treinamento)\n\n")
            
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
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Visualizações salvas em: evaluation_results/\n")
            f.write("=" * 70 + "\n")
        
        print(f"  ✓ Relatório salvo: {report_path}")


def main():
    print("=" * 60)
    print("🦋 ANÁLISE DE ERRO - VISUALIZAÇÕES OTIMIZADAS")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    
    # Gerar visualizações OTIMIZADAS
    evaluator.plot_performance_summary()  # Dashboard principal
    evaluator.plot_top_bottom_classes()   # Melhores e piores
    evaluator.plot_confusion_matrix_aggregated()  # Confusão simplificada
    
    # Análise de erro
    evaluator.analyze_errors()
    
    # Relatório final
    evaluator.generate_summary_report()
    
    print("\n" + "=" * 60)
    print("✅ ANÁLISE COMPLETA!")
    print("=" * 60)
    print("\nArquivos gerados (OTIMIZADOS):")
    print("  • performance_dashboard.png     (visão geral)")
    print("  • top_bottom_classes.png        (melhores/piores)")
    print("  • confusion_matrix_simplified.png (confusão agregada)")
    print("  • error_analysis.txt            (detalhes)")
    print("  • summary_report.txt            (resumo)")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()