#!/usr/bin/env python3
"""
Predict Butterfly Species - Inference Script
Classifica espécie de borboleta a partir de uma imagem
"""

import cv2
import numpy as np
import joblib
import sys
from pathlib import Path

class ButterflyPredictor:
    def __init__(self, model_path='models/svm_model.pkl', 
                 encoder_path='models/label_encoder.pkl'):
        """Carrega modelo treinado e label encoder"""
        print("📂 Carregando modelo...")
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        print(f"✓ Modelo carregado: {len(self.label_encoder.classes_)} espécies")
        print(f"  Espécies conhecidas: {', '.join(self.label_encoder.classes_[:5])}...\n")
    
    def preprocess_image(self, img):
        """
        Replica o pré-processamento do preprocess.cpp
        """
        # 1. Redimensionar para 224x224
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Bilateral filter (denoise)
        denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)
        
        # 3. CLAHE (equalização adaptativa)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        
        # 4. Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(result, -1, kernel)
        
        # Blend: 70% equalizado + 30% sharpened
        final = cv2.addWeighted(result, 0.7, sharpened, 0.3, 0)
        
        return final
    
    def extract_hog_features(self, img):
        """
        Extrai features HOG (replica feature_extraction.cpp)
        """
        # Converter para grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar para 64x128 (winSize do HOG)
        resized = cv2.resize(gray, (64, 128))
        
        # HOGDescriptor com mesmos parâmetros do C++
        hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
        features = hog.compute(resized)
        return features.flatten()
    
    def extract_lbp_features(self, img):
        """
        Extrai features LBP (replica feature_extraction.cpp)
        """
        # Converter para grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar para 128x128
        resized = cv2.resize(gray, (128, 128))
        
        # LBP 3x3
        lbp = np.zeros_like(resized)
        
        for i in range(1, resized.shape[0] - 1):
            for j in range(1, resized.shape[1] - 1):
                center = resized[i, j]
                
                # Compara 8 vizinhos (sentido horário)
                code = 0
                code |= (resized[i-1, j-1] >= center) << 7
                code |= (resized[i-1, j]   >= center) << 6
                code |= (resized[i-1, j+1] >= center) << 5
                code |= (resized[i, j+1]   >= center) << 4
                code |= (resized[i+1, j+1] >= center) << 3
                code |= (resized[i+1, j]   >= center) << 2
                code |= (resized[i+1, j-1] >= center) << 1
                code |= (resized[i, j-1]   >= center) << 0
                
                lbp[i, j] = code
        
        # Calcular histograma (256 bins)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        
        return hist.astype(np.float32)
    
    def extract_combined_features(self, img):
        """Combina HOG + LBP"""
        hog = self.extract_hog_features(img)
        lbp = self.extract_lbp_features(img)
        
        combined = np.concatenate([hog, lbp])
        
        # Normalização L2 (mesmo do C++)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def predict(self, image_path, show_confidence=True):
        """
        Prediz espécie de borboleta a partir de uma imagem
        
        Args:
            image_path: Caminho da imagem
            show_confidence: Se True, mostra probabilidades das top-3 classes
        
        Returns:
            Tupla (espécie_predita, confiança)
        """
        # Carregar imagem
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        print(f"🖼️  Analisando: {Path(image_path).name}")
        print(f"   Tamanho original: {img.shape[1]}x{img.shape[0]}")
        
        # Pipeline completo
        img_processed = self.preprocess_image(img)
        features = self.extract_combined_features(img_processed)
        
        # Predição
        prediction = self.model.predict([features])[0]
        species = self.label_encoder.inverse_transform([prediction])[0]
        
        # Confiança (probabilidades)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba([features])[0]
            confidence = proba[prediction]
            
            if show_confidence:
                # Top-3 classes mais prováveis
                top3_indices = np.argsort(proba)[-3:][::-1]
                print(f"\n📊 Confiança da predição:")
                for i, idx in enumerate(top3_indices):
                    label = self.label_encoder.classes_[idx]
                    prob = proba[idx]
                    symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    print(f"   {symbol} {label:30s}: {prob*100:5.2f}%")
        else:
            confidence = None
        
        return species, confidence
    
    def predict_batch(self, image_dir, output_csv='predictions.csv'):
        """
        Prediz espécies para todas as imagens de um diretório
        
        Args:
            image_dir: Diretório com imagens
            output_csv: Arquivo de saída com predições
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            raise ValueError(f"Diretório não encontrado: {image_dir}")
        
        # Encontrar todas as imagens
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(image_dir.glob(f'*{ext}'))
            images.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not images:
            print(f"⚠️  Nenhuma imagem encontrada em {image_dir}")
            return
        
        print(f"\n🦋 Processando {len(images)} imagens...")
        print("=" * 60)
        
        results = []
        
        for i, img_path in enumerate(images, 1):
            try:
                species, confidence = self.predict(str(img_path), show_confidence=False)
                
                results.append({
                    'filename': img_path.name,
                    'predicted_species': species,
                    'confidence': confidence if confidence else 'N/A'
                })
                
                conf_str = f"({confidence*100:.2f}%)" if confidence else ""
                print(f"[{i}/{len(images)}] {img_path.name:30s} → {species} {conf_str}")
                
            except Exception as e:
                print(f"❌ Erro em {img_path.name}: {e}")
                results.append({
                    'filename': img_path.name,
                    'predicted_species': 'ERROR',
                    'confidence': 0
                })
        
        # Salvar resultados em CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        print("\n" + "=" * 60)
        print(f"✅ Predições salvas em: {output_csv}")
        
        # Estatísticas
        species_count = df['predicted_species'].value_counts()
        print(f"\n📊 Distribuição de espécies preditas:")
        for species, count in species_count.head(5).items():
            print(f"   {species:30s}: {count} imagens")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🦋 Butterfly Species Classifier - Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  
  # Classificar uma única imagem
  python predict_butterfly.py --image dataset/test/Image_1.jpg
  
  # Classificar todas as imagens de um diretório
  python predict_butterfly.py --batch dataset/test
  
  # Avaliar modelo no conjunto de teste (com ground truth)
  python predict_butterfly.py --evaluate dataset/test dataset/Testing_set.csv
        """
    )
    
    parser.add_argument('--image', type=str, help='Caminho de uma imagem')
    parser.add_argument('--batch', type=str, help='Diretório com múltiplas imagens')
    parser.add_argument('--evaluate', nargs=2, metavar=('IMAGE_DIR', 'CSV_FILE'),
                       help='Avalia modelo com ground truth (diretório + CSV)')
    parser.add_argument('--model', type=str, default='models/svm_model.pkl',
                       help='Caminho do modelo (padrão: models/svm_model.pkl)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Arquivo de saída para predições (padrão: predictions.csv)')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not any([args.image, args.batch, args.evaluate]):
        parser.print_help()
        sys.exit(1)
    
    # Inicializar preditor
    try:
        predictor = ButterflyPredictor(model_path=args.model)
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
        print("\nCertifique-se de ter treinado o modelo primeiro:")
        print("  make train")
        sys.exit(1)
    
    print("=" * 60)
    
    # Modo: Imagem única
    if args.image:
        try:
            species, confidence = predictor.predict(args.image)
            print("\n" + "=" * 60)
            print(f"🦋 RESULTADO: {species}")
            if confidence:
                print(f"   Confiança: {confidence*100:.2f}%")
            print("=" * 60)
        except Exception as e:
            print(f"❌ Erro: {e}")
            sys.exit(1)
    
    # Modo: Batch
    elif args.batch:
        try:
            predictor.predict_batch(args.batch, args.output)
        except Exception as e:
            print(f"❌ Erro: {e}")
            sys.exit(1)
    
    # Modo: Avaliação com ground truth
    elif args.evaluate:
        image_dir, csv_file = args.evaluate
        
        try:
            # Carregar ground truth
            import pandas as pd
            df_true = pd.read_csv(csv_file)
            
            # Criar mapa filename -> label
            if 'label' in df_true.columns:
                true_labels = dict(zip(df_true['filename'], df_true['label']))
            else:
                print("⚠️  CSV não tem coluna 'label', pulando avaliação")
                predictor.predict_batch(image_dir, args.output)
                sys.exit(0)
            
            # Predizer todas as imagens
            image_dir_path = Path(image_dir)
            images = list(image_dir_path.glob('*.jpg')) + \
                    list(image_dir_path.glob('*.jpeg')) + \
                    list(image_dir_path.glob('*.png'))
            
            results = []
            correct = 0
            total = 0
            
            print(f"\n🔍 Avaliando modelo no conjunto de teste...")
            print("=" * 60)
            
            for img_path in images:
                if img_path.name not in true_labels:
                    continue
                
                try:
                    species_pred, confidence = predictor.predict(str(img_path), show_confidence=False)
                    species_true = true_labels[img_path.name]
                    
                    is_correct = (species_pred == species_true)
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    symbol = "✅" if is_correct else "❌"
                    print(f"{symbol} {img_path.name:30s} | True: {species_true:20s} | Pred: {species_pred}")
                    
                    results.append({
                        'filename': img_path.name,
                        'true_species': species_true,
                        'predicted_species': species_pred,
                        'correct': is_correct,
                        'confidence': confidence if confidence else 'N/A'
                    })
                    
                except Exception as e:
                    print(f"❌ Erro em {img_path.name}: {e}")
            
            # Salvar resultados
            df_results = pd.DataFrame(results)
            df_results.to_csv(args.output, index=False)
            
            # Estatísticas
            accuracy = correct / total if total > 0 else 0
            
            print("\n" + "=" * 60)
            print(f"📊 RESULTADO DA AVALIAÇÃO:")
            print(f"   Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
            print(f"   Resultados salvos em: {args.output}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()