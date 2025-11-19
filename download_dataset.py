#!/usr/bin/env python3
"""
Download automático do Butterfly Dataset do Kaggle
Uso: python3 download_dataset.py
Requer: KAGGLE_USERNAME e KAGGLE_KEY como variáveis de ambiente
"""

import os
import sys
import shutil
from pathlib import Path

def check_credentials():
    """Verifica se credenciais do Kaggle estão configuradas"""
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    
    if not username or not key:
        print("❌ Credenciais do Kaggle não encontradas!\n")
        print("Configure as variáveis de ambiente:")
        print("  export KAGGLE_USERNAME='seu_username'")
        print("  export KAGGLE_KEY='sua_key'\n")
        print("Obtenha suas credenciais em: https://www.kaggle.com/settings")
        return False
    
    print(f"✓ Credenciais encontradas (Username: {username})")
    return True

def download_dataset():
    """Baixa dataset usando API do Kaggle"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("\n=== Baixando dataset ===")
        print("Dataset: phucthaiv02/butterfly-image-classification\n")
        
        api = KaggleApi()
        api.authenticate()
        
        temp_dir = Path('./temp_download')
        temp_dir.mkdir(exist_ok=True)
        
        print("⏳ Baixando... (pode levar alguns minutos)")
        api.dataset_download_files(
            'phucthaiv02/butterfly-image-classification',
            path=str(temp_dir),
            unzip=True
        )
        
        print("✓ Download concluído!")
        return temp_dir
        
    except ImportError:
        print("❌ Kaggle API não instalada!")
        print("Execute: pip install kaggle")
        return None
    except Exception as e:
        print(f"❌ Erro no download: {e}")
        return None

def organize_dataset(temp_dir):
    """Organiza dataset na estrutura final"""
    print("\n=== Organizando dataset ===")
    
    # Procura pelo diretório baixado
    possible_roots = [
        temp_dir / 'butterfly-image-classification',
        temp_dir
    ]
    
    dataset_root = None
    for root in possible_roots:
        if root.exists():
            subdirs = [d.name for d in root.iterdir() if d.is_dir()]
            if 'train' in subdirs or 'test' in subdirs:
                dataset_root = root
                break
    
    if not dataset_root:
        print("❌ Estrutura do dataset não encontrada!")
        return False
    
    print(f"✓ Dataset encontrado em: {dataset_root}")
    
    # Cria estrutura final
    final_dir = Path('./dataset')
    final_dir.mkdir(exist_ok=True)
    
    # Move diretórios (train, test, val)
    for split in ['train', 'test', 'val', 'valid']:
        src = dataset_root / split
        if src.exists():
            dst = final_dir / ('val' if split == 'valid' else split)
            
            if dst.exists():
                shutil.rmtree(dst)
            
            shutil.copytree(src, dst)
            print(f"✓ Copiado: {split}/ -> dataset/{dst.name}/")
    
    # Remove temporários
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Estatísticas
    print("\n=== Estatísticas ===")
    total = 0
    for split_dir in sorted(final_dir.iterdir()):
        if split_dir.is_dir():
            species = [d for d in split_dir.iterdir() if d.is_dir()]
            n_species = len(species)
            n_images = sum(len(list(d.glob('*.jpg'))) + len(list(d.glob('*.png'))) 
                          for d in species)
            print(f"  {split_dir.name}: {n_species} espécies, {n_images} imagens")
            total += n_images
    
    print(f"\n📊 Total: {total} imagens")
    return True

def main():
    print("=" * 60)
    print("  🦋 BUTTERFLY DATASET DOWNLOADER")
    print("=" * 60)
    print()
    
    # Verifica credenciais
    if not check_credentials():
        sys.exit(1)
    
    # Baixa dataset
    temp_dir = download_dataset()
    if not temp_dir:
        sys.exit(1)
    
    # Organiza estrutura
    if not organize_dataset(temp_dir):
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  ✅ DATASET PRONTO!")
    print("=" * 60)
    print(f"\n📂 Localização: {Path('./dataset').absolute()}")
    print("\n🚀 Próximo passo:")
    print("   make compile && make preprocess")

if __name__ == "__main__":
    main()