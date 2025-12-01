#!/usr/bin/env python3
"""
Predict Butterfly
Predição rápida em batch com cache
"""

import cv2
import numpy as np
import joblib
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

class FastButterflyPredictor:
    def __init__(self, model_path='models/best_model.pkl', 
                 encoder_path='models/label_encoder.pkl',
                 scaler_path='models/scaler.pkl'):
        print("📂 Carregando modelo...")
        
        # Fallback para modelos antigos
        if not Path(model_path).exists():
            model_path = 'models/svm_model.pkl'
        
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.scaler = joblib.load(scaler_path)
        
        print(f"  Modelo: {Path(model_path).name}")
        print(f"  Classes: {len(self.label_encoder.classes_)}\n")
    
    def preprocess_image(self, img_bgr):
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr
        
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LINEAR)
        equalized = cv2.equalizeHist(resized)
        blended = cv2.addWeighted(equalized, 0.7, resized, 0.3, 0)
        
        return blended
    
    def extract_hog_compact(self, img):
        resized = cv2.resize(img, (64, 128), interpolation=cv2.INTER_LINEAR)
        hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        features = hog.compute(resized).flatten()
        # Redução: média em blocos de 4
        compact = [features[i:i+4].mean() for i in range(0, len(features), 4)]
        return np.array(compact, dtype=np.float32)
    
    def extract_lbp_compact(self, img):
        resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
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
        
        hist, _ = np.histogram(lbp.ravel(), bins=128, range=(0, 256))
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        
        return hist.astype(np.float32)
    
    def extract_color_compact(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        return hist.flatten()
    
    def extract_features(self, img_bgr):
        gray = self.preprocess_image(img_bgr)
        hog = self.extract_hog_compact(gray)
        lbp = self.extract_lbp_compact(gray)
        color = self.extract_color_compact(img_bgr)
        
        combined = np.concatenate([hog, lbp, color])
        norm = np.linalg.norm(combined)
        if norm > 1e-7:
            combined = combined / norm
        
        return combined
    
    def predict_single(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Não carregou: {image_path}")
        
        features = self.extract_features(img)
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        species = self.label_encoder.inverse_transform([prediction])[0]
        
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            confidence = proba[prediction]
        elif hasattr(self.model, 'decision_function'):
            # LinearSVC usa decision function
            decision = self.model.decision_function(features_scaled)[0]
            confidence = 1.0 / (1.0 + np.exp(-decision.max()))
        
        return species, confidence
    
    def predict_batch_parallel(self, image_dir, output_csv='predictions.csv', max_workers=4):
        image_dir = Path(image_dir)
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in extensions:
            images.extend(image_dir.glob(f'*{ext}'))
            images.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not images:
            print(f"⚠️  Nenhuma imagem em {image_dir}")
            return
        
        print(f"\n🦋 Processando {len(images)} imagens (paralelo)...\n")
        
        results = []
        errors = 0
        start_time = time.time()
        
        def process_image(args):
            i, img_path = args
            try:
                species, confidence = self.predict_single(str(img_path))
                conf_pct = f"{confidence*100:.2f}%" if confidence else "N/A"
                return {
                    'idx': i,
                    'filename': img_path.name,
                    'predicted_species': species,
                    'confidence': conf_pct,
                    'error': None
                }
            except Exception as e:
                return {
                    'idx': i,
                    'filename': img_path.name,
                    'predicted_species': 'ERROR',
                    'confidence': '0%',
                    'error': str(e)
                }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(process_image, enumerate(images, 1)):
                results.append(result)
                
                if result['error']:
                    errors += 1
                    print(f"[{result['idx']:4d}/{len(images)}] ❌ {result['filename']}")
                else:
                    print(f"[{result['idx']:4d}/{len(images)}] {result['filename']:35s} → {result['predicted_species']:30s} ({result['confidence']})")
        
        elapsed = time.time() - start_time
        
        # Salvar
        import pandas as pd
        df = pd.DataFrame(results)
        df = df.drop(columns=['idx', 'error'])
        df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"✅ Completo em {elapsed:.1f}s")
        print(f"   Arquivo: {output_csv}")
        print(f"   Total: {len(images)} | Sucesso: {len(images)-errors} | Erros: {errors}")
        print(f"{'='*70}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='🦋 Butterfly Classifier')
    parser.add_argument('--image', type=str)
    parser.add_argument('--batch', type=str)
    parser.add_argument('--evaluate', nargs=2, metavar=('DIR', 'CSV'))
    parser.add_argument('--output', type=str, default='predictions.csv')
    parser.add_argument('--workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if not any([args.image, args.batch, args.evaluate]):
        parser.print_help()
        sys.exit(1)
    
    try:
        predictor = FastButterflyPredictor()
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        sys.exit(1)
    
    if args.image:
        try:
            species, confidence = predictor.predict_single(args.image)
            print(f"\n{'='*70}")
            print(f"🦋 PREDIÇÃO: {species}")
            if confidence:
                print(f"   Confiança: {confidence*100:.2f}%")
            print(f"{'='*70}")
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            sys.exit(1)
    
    elif args.batch:
        try:
            predictor.predict_batch_parallel(args.batch, args.output, args.workers)
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            sys.exit(1)
    
    elif args.evaluate:
        import pandas as pd
        image_dir, csv_file = args.evaluate
        
        df_gt = pd.read_csv(csv_file)
        if 'label' not in df_gt.columns:
            print("⚠️  CSV sem 'label', modo batch normal")
            predictor.predict_batch_parallel(image_dir, args.output, args.workers)
            sys.exit(0)
        
        true_labels = dict(zip(df_gt['filename'], df_gt['label']))
        
        image_dir_path = Path(image_dir)
        images = list(image_dir_path.glob('*.jpg')) + list(image_dir_path.glob('*.jpeg'))
        
        results = []
        correct = 0
        total = 0
        
        print(f"\n🔍 Avaliando {len(images)} imagens...\n")
        
        for img_path in images:
            if img_path.name not in true_labels:
                continue
            
            try:
                pred_species, confidence = predictor.predict_single(str(img_path))
                true_species = true_labels[img_path.name]
                
                is_correct = (pred_species == true_species)
                if is_correct:
                    correct += 1
                total += 1
                
                symbol = "✅" if is_correct else "❌"
                conf_pct = f"{confidence*100:.2f}%" if confidence else "N/A"
                
                print(f"{symbol} {img_path.name:35s} | True: {true_species:25s} | Pred: {pred_species:25s}")
                
                results.append({
                    'filename': img_path.name,
                    'true_species': true_species,
                    'predicted_species': pred_species,
                    'correct': is_correct,
                    'confidence': conf_pct
                })
            except Exception as e:
                print(f"❌ {img_path.name}: {e}")
        
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n{'='*70}")
        print(f"📊 ACCURACY: {accuracy*100:.2f}% ({correct}/{total})")
        print(f"   Arquivo: {args.output}")
        print(f"{'='*70}")

if __name__ == '__main__':
    main()