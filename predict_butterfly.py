#!/usr/bin/env python3
"""
Predict Butterfly V2 - CORRIGIDO
Usa o mesmo pipeline do treino (com scaler!)
"""

import cv2
import numpy as np
import joblib
import sys
from pathlib import Path

class ButterflyPredictor:
    def __init__(self, model_path='models/svm_model.pkl', 
                 encoder_path='models/label_encoder.pkl',
                 scaler_path='models/scaler.pkl'):
        """Carrega modelo, encoder E scaler"""
        print("📂 Carregando modelo...")
        
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.scaler = joblib.load(scaler_path)  # CRÍTICO!
            
            print(f"✓ Modelo: {model_path}")
            print(f"✓ Scaler: {scaler_path}")
            print(f"✓ Classes: {len(self.label_encoder.classes_)}")
            print(f"  {', '.join(self.label_encoder.classes_[:3])}...\n")
            
        except FileNotFoundError as e:
            print(f"❌ ERRO: {e}")
            print("\nVerifique se você treinou o modelo:")
            print("  python3 train_classifier_v2.py")
            raise
    
    def preprocess_image(self, img_bgr):
        """
        Pré-processamento IDÊNTICO ao C++
        """
        # Converter para grayscale
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr
        
        # Resize
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Equalização
        equalized = cv2.equalizeHist(resized)
        
        # Blend 70/30
        blended = cv2.addWeighted(equalized, 0.7, resized, 0.3, 0)
        
        return blended
    
    def extract_hog_features(self, img):
        """HOG features"""
        resized = cv2.resize(img, (64, 128), interpolation=cv2.INTER_LINEAR)
        
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
        """LBP features"""
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        lbp = np.zeros_like(resized)
        
        for i in range(1, resized.shape[0] - 1):
            for j in range(1, resized.shape[1] - 1):
                center = resized[i, j]
                
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
        
        # Histograma normalizado
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        
        return hist.astype(np.float32)
    
    def extract_color_histogram(self, img_bgr):
        """Color histogram (HSV)"""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        hist = cv2.calcHist(
            [hsv], 
            [0, 1, 2],  # H, S, V
            None, 
            [16, 8, 8],  # bins
            [0, 180, 0, 256, 0, 256]  # ranges
        )
        
        # Normalizar
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        
        return hist.flatten()
    
    def extract_combined_features(self, img_bgr):
        """Combina HOG + LBP + Color"""
        # Processar
        gray = self.preprocess_image(img_bgr)
        
        # Extrair
        hog = self.extract_hog_features(gray)
        lbp = self.extract_lbp_features(gray)
        color = self.extract_color_histogram(img_bgr)
        
        # Combinar
        combined = np.concatenate([hog, lbp, color])
        
        # Normalizar (L2)
        norm = np.linalg.norm(combined)
        if norm > 1e-7:
            combined = combined / norm
        
        return combined
    
    def predict(self, image_path, show_confidence=True):
        """Prediz espécie"""
        # Carregar imagem em BGR
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Não foi possível carregar: {image_path}")
        
        print(f"🖼️  Analisando: {Path(image_path).name}")
        print(f"   Tamanho: {img.shape[1]}x{img.shape[0]}")
        
        # Extrair features
        features = self.extract_combined_features(img)
        
        # CRÍTICO: Aplicar o mesmo scaler do treino
        features_scaled = self.scaler.transform([features])
        
        # Predição
        prediction = self.model.predict(features_scaled)[0]
        species = self.label_encoder.inverse_transform([prediction])[0]
        
        # Confiança
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            confidence = proba[prediction]
            
            if show_confidence:
                # Top-5
                top5_indices = np.argsort(proba)[-5:][::-1]
                print(f"\n📊 Top-5 predições:")
                symbols = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for i, idx in enumerate(top5_indices):
                    label = self.label_encoder.classes_[idx]
                    prob = proba[idx]
                    print(f"   {symbols[i]} {label:30s}: {prob*100:6.2f}%")
        
        return species, confidence
    
    def predict_batch(self, image_dir, output_csv='predictions.csv'):
        """Batch prediction"""
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            raise ValueError(f"Diretório não encontrado: {image_dir}")
        
        # Encontrar imagens
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in extensions:
            images.extend(image_dir.glob(f'*{ext}'))
            images.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not images:
            print(f"⚠️  Nenhuma imagem em {image_dir}")
            return
        
        print(f"\n🦋 Processando {len(images)} imagens...\n")
        
        results = []
        errors = 0
        
        for i, img_path in enumerate(images, 1):
            try:
                species, confidence = self.predict(str(img_path), show_confidence=False)
                
                conf_pct = f"{confidence*100:.2f}%" if confidence else "N/A"
                
                results.append({
                    'filename': img_path.name,
                    'predicted_species': species,
                    'confidence': conf_pct
                })
                
                print(f"[{i:4d}/{len(images)}] {img_path.name:35s} → {species:30s} ({conf_pct})")
                
            except Exception as e:
                print(f"[{i:4d}/{len(images)}] ❌ {img_path.name}: {e}")
                errors += 1
                results.append({
                    'filename': img_path.name,
                    'predicted_species': 'ERROR',
                    'confidence': '0%'
                })
        
        # Salvar CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"✅ Predições salvas: {output_csv}")
        print(f"   Total: {len(images)} | Sucesso: {len(images)-errors} | Erros: {errors}")
        
        # Distribuição
        if len(results) > 0:
            species_count = df['predicted_species'].value_counts()
            print(f"\n📊 Top-5 espécies preditas:")
            for species, count in species_count.head(5).items():
                print(f"   {species:30s}: {count:4d} imagens")
        
        print(f"{'='*70}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🦋 Butterfly Classifier V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  
  # Uma imagem
  python predict_butterfly_v2.py --image dataset/train/Image_1.jpg
  
  # Várias imagens
  python predict_butterfly_v2.py --batch dataset/test
  
  # Avaliar com ground truth
  python predict_butterfly_v2.py --evaluate dataset/train dataset/Training_set.csv
        """
    )
    
    parser.add_argument('--image', type=str)
    parser.add_argument('--batch', type=str)
    parser.add_argument('--evaluate', nargs=2, metavar=('DIR', 'CSV'))
    parser.add_argument('--output', type=str, default='predictions.csv')
    
    args = parser.parse_args()
    
    if not any([args.image, args.batch, args.evaluate]):
        parser.print_help()
        sys.exit(1)
    
    # Carregar preditor
    try:
        predictor = ButterflyPredictor()
    except Exception as e:
        sys.exit(1)
    
    print("=" * 70)
    
    # Modo: Imagem única
    if args.image:
        try:
            species, confidence = predictor.predict(args.image)
            print(f"\n{'='*70}")
            print(f"🦋 PREDIÇÃO: {species}")
            if confidence:
                print(f"   Confiança: {confidence*100:.2f}%")
            print(f"{'='*70}")
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            sys.exit(1)
    
    # Modo: Batch
    elif args.batch:
        try:
            predictor.predict_batch(args.batch, args.output)
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Modo: Avaliação
    elif args.evaluate:
        image_dir, csv_file = args.evaluate
        
        try:
            import pandas as pd
            df_gt = pd.read_csv(csv_file)
            
            if 'label' not in df_gt.columns:
                print("⚠️  CSV sem coluna 'label', fazendo batch normal")
                predictor.predict_batch(image_dir, args.output)
                sys.exit(0)
            
            true_labels = dict(zip(df_gt['filename'], df_gt['label']))
            
            # Processar
            image_dir_path = Path(image_dir)
            images = list(image_dir_path.glob('*.jpg')) + \
                    list(image_dir_path.glob('*.jpeg')) + \
                    list(image_dir_path.glob('*.png'))
            
            results = []
            correct = 0
            total = 0
            
            print(f"\n🔍 Avaliando {len(images)} imagens...\n")
            
            for img_path in images:
                if img_path.name not in true_labels:
                    continue
                
                try:
                    pred_species, confidence = predictor.predict(str(img_path), show_confidence=False)
                    true_species = true_labels[img_path.name]
                    
                    is_correct = (pred_species == true_species)
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    symbol = "✅" if is_correct else "❌"
                    conf_pct = f"{confidence*100:.2f}%" if confidence else "N/A"
                    
                    print(f"{symbol} {img_path.name:35s} | "
                          f"True: {true_species:25s} | "
                          f"Pred: {pred_species:25s} ({conf_pct})")
                    
                    results.append({
                        'filename': img_path.name,
                        'true_species': true_species,
                        'predicted_species': pred_species,
                        'correct': is_correct,
                        'confidence': conf_pct
                    })
                    
                except Exception as e:
                    print(f"❌ {img_path.name}: {e}")
            
            # Salvar
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            
            # Estatísticas
            accuracy = correct / total if total > 0 else 0
            
            print(f"\n{'='*70}")
            print(f"📊 RESULTADO FINAL:")
            print(f"   Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
            print(f"   Arquivo: {args.output}")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()