#!/usr/bin/env python3
"""
Download Butterfly Dataset from Kaggle
Baixa e organiza o dataset de classificação de borboletas
"""

import os
import zipfile
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """Verifica credenciais Kaggle"""
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    
    if not username or not key:
        print("❌ ERRO: Credenciais Kaggle não configuradas!")
        print("\nConfigure com:")
        print("  export KAGGLE_USERNAME='seu_username'")
        print("  export KAGGLE_KEY='sua_key'")
        print("\nObtenha em: https://www.kaggle.com/settings")
        exit(1)
    
    # Criar ~/.kaggle/kaggle.json
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    with open(kaggle_json, 'w') as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}\n')
    
    # Permissões corretas
    kaggle_json.chmod(0o600)
    
    print("✓ Credenciais Kaggle configuradas")

def download_dataset():
    """Baixa dataset do Kaggle"""
    import kaggle
    
    dataset_slug = 'phucthaiv02/butterfly-image-classification'
    download_path = 'dataset_temp'
    
    print(f"\n📥 Baixando dataset: {dataset_slug}")
    print("Isso pode demorar alguns minutos...")
    
    # Limpar diretório temporário se existir
    if os.path.exists(download_path):
        shutil.rmtree(download_path)
    
    os.makedirs(download_path, exist_ok=True)
    
    # Download
    kaggle.api.dataset_download_files(
        dataset_slug,
        path=download_path,
        unzip=True
    )
    
    print("✓ Download concluído")
    
    return download_path

def organize_dataset(temp_path):
    """Organiza dataset na estrutura correta"""
    print("\n📁 Organizando estrutura do dataset...")
    
    final_path = Path('dataset')
    
    # Remover dataset antigo se existir
    if final_path.exists():
        print("  Removendo dataset antigo...")
        shutil.rmtree(final_path)
    
    # Mover para estrutura final
    temp_path_obj = Path(temp_path)
    
    # O dataset do Kaggle já vem com train/ e test/
    # Cada um contém pastas de espécies
    
    if (temp_path_obj / 'train').exists():
        print("  ✓ Estrutura detectada: train/")
        shutil.move(str(temp_path_obj), str(final_path))
    else:
        # Caso esteja em outra estrutura, adaptar
        print("  ⚠️  Estrutura diferente, adaptando...")
        final_path.mkdir(exist_ok=True)
        
        for item in temp_path_obj.iterdir():
            shutil.move(str(item), str(final_path / item.name))
    
    # Limpar arquivos temporários
    if temp_path_obj.exists():
        shutil.rmtree(temp_path_obj, ignore_errors=True)
    
    print("✓ Dataset organizado")
    
    return final_path

def verify_structure(dataset_path):
    """Verifica estrutura do dataset"""
    print("\n🔍 Verificando estrutura...")
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print("❌ ERRO: dataset/ não encontrado!")
        return False
    
    # Verificar train e test
    train_path = dataset_path / 'train'
    test_path = dataset_path / 'test'
    
    if not train_path.exists():
        print("❌ ERRO: dataset/train/ não encontrado!")
        return False
    
    if not test_path.exists():
        print("⚠️  AVISO: dataset/test/ não encontrado (opcional)")
    
    # Contar espécies
    species_train = [d.name for d in train_path.iterdir() if d.is_dir()]
    num_species = len(species_train)
    
    print(f"\n✓ Estrutura verificada:")
    print(f"  - Train: {train_path}")
    print(f"  - Test:  {test_path if test_path.exists() else 'N/A'}")
    print(f"  - Espécies encontradas: {num_species}")
    
    if num_species > 0:
        print(f"\n  Primeiras 5 espécies:")
        for species in species_train[:5]:
            num_images = len(list((train_path / species).glob('*.*')))
            print(f"    - {species}: {num_images} imagens")
    
    # Contar total de imagens
    total_train = sum(1 for _ in train_path.rglob('*.jpg')) + \
                  sum(1 for _ in train_path.rglob('*.jpeg')) + \
                  sum(1 for _ in train_path.rglob('*.png'))
    
    total_test = 0
    if test_path.exists():
        total_test = sum(1 for _ in test_path.rglob('*.jpg')) + \
                     sum(1 for _ in test_path.rglob('*.jpeg')) + \
                     sum(1 for _ in test_path.rglob('*.png'))
    
    print(f"\n  Total de imagens:")
    print(f"    - Train: {total_train}")
    print(f"    - Test:  {total_test}")
    print(f"    - Total: {total_train + total_test}")
    
    return True

def main():
    print("=" * 60)
    print("🦋 BUTTERFLY DATASET DOWNLOADER")
    print("=" * 60)
    
    # 1. Setup credenciais
    setup_kaggle_credentials()
    
    # 2. Download
    temp_path = download_dataset()
    
    # 3. Organizar
    final_path = organize_dataset(temp_path)
    
    # 4. Verificar
    if verify_structure(final_path):
        print("\n" + "=" * 60)
        print("✅ DATASET PRONTO!")
        print("=" * 60)
        print("\nPróximo passo: make preprocess")
    else:
        print("\n❌ Erro na estrutura do dataset!")
        exit(1)

if __name__ == '__main__':
    main()