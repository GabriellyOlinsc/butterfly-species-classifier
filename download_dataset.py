"""
Script para baixar dataset de borboletas do Kaggle
Funciona com: .env local, variáveis de ambiente, ou interativo
Requer: pip install kaggle opendatasets python-dotenv
"""

import os
import sys
import shutil
from pathlib import Path

def load_env_file():
    """Carrega credenciais do arquivo .env se existir"""
    try:
        from dotenv import load_dotenv
        
        env_file = Path('.env')
        if env_file.exists():
            load_dotenv(env_file)
            print("✓ Arquivo .env encontrado e carregado")
            return True
        return False
    except ImportError:
        print("⚠ python-dotenv não instalado (execute: pip install python-dotenv)")
        return False

def setup_kaggle_credentials():
    """Configura credenciais do Kaggle (suporta .env, variáveis de ambiente, ou interativo)"""
    print("=== Verificando Credenciais do Kaggle ===\n")
    
    # Tenta carregar do .env primeiro
    env_loaded = load_env_file()
    
    # Verifica se credenciais estão disponíveis
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    
    if username and key:
        print("✓ Credenciais encontradas!")
        source = "arquivo .env" if env_loaded else "variáveis de ambiente"
        print(f"  Origem: {source}")
        print(f"  Username: {username}")
        print(f"  Key: {'*' * 20}{key[-4:]}")
        return True
    
    print("⚠ Credenciais não encontradas.\n")
    print("=" * 70)
    print("  CONFIGURAÇÃO DE CREDENCIAIS")
    print("=" * 70)
    
    print("\n📝 PASSO 1: Obter suas credenciais do Kaggle")
    print("   1. Acesse: https://www.kaggle.com/settings")
    print("   2. Role até a seção 'API'")
    print("   3. Clique em 'Create New API Token'")
    print("   4. Isso baixará o arquivo kaggle.json")
    print("   5. Abra o arquivo e copie o username e key")
    
    print("\n🔐 PASSO 2: Configurar credenciais")
    print("\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   📌 OPÇÃO A - Arquivo .env (RECOMENDADO para uso local)")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   1. Copie o arquivo .env.example:")
    print("      cp .env.example .env")
    print("   2. Edite o arquivo .env e preencha suas credenciais:")
    print("      KAGGLE_USERNAME=seu_username")
    print("      KAGGLE_KEY=sua_key")
    print("   3. Execute este script novamente")
    
    print("\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   📌 OPÇÃO B - GitHub Codespaces (Secrets)")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   1. Vá em: Settings > Secrets and variables > Codespaces")
    print("   2. Adicione dois secrets:")
    print("      - KAGGLE_USERNAME = seu_username")
    print("      - KAGGLE_KEY = sua_key")
    print("   3. Reinicie o Codespace")
    
    print("\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   📌 OPÇÃO C - Variáveis de ambiente (temporário)")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   Execute no terminal:")
    print("      export KAGGLE_USERNAME='seu_username'")
    print("      export KAGGLE_KEY='sua_key'")
    
    print("\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   📌 OPÇÃO D - Fornecer agora (apenas para esta execução)")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    response = input("\nDeseja fornecer as credenciais agora? (s/n): ").strip().lower()
    
    if response == 's':
        username = input("Kaggle Username: ").strip()
        key = input("Kaggle Key: ").strip()
        
        if username and key:
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
            print("\n✓ Credenciais configuradas para esta sessão!")
            print("⚠ NOTA: Essas credenciais serão perdidas ao fechar o terminal.")
            print("   Para permanência, use a OPÇÃO A (arquivo .env).")
            
            # Oferece salvar no .env
            save_env = input("\nDeseja salvar no arquivo .env? (s/n): ").strip().lower()
            if save_env == 's':
                env_content = f"""# Credenciais da API do Kaggle
KAGGLE_USERNAME={username}
KAGGLE_KEY={key}
"""
                with open('.env', 'w') as f:
                    f.write(env_content)
                print("✓ Credenciais salvas em .env")
                print("⚠ IMPORTANTE: Não commite o arquivo .env no Git!")
            
            return True
        else:
            print("✗ Username ou Key vazios!")
            return False
    
    print("\n❌ Não é possível continuar sem credenciais.")
    print("Configure as credenciais e execute novamente.")
    return False

def download_with_kaggle_api():
    """Baixa usando API oficial do Kaggle"""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("\n=== Baixando dataset com Kaggle API ===")
        print("Dataset: phucthaiv02/butterfly-image-classification\n")
        
        # Autentica usando variáveis de ambiente
        api = KaggleApi()
        api.authenticate()
        
        # Cria diretório temporário
        temp_dir = Path('./dataset_temp')
        temp_dir.mkdir(exist_ok=True)
        
        # Baixa o dataset
        print("⏳ Baixando... (isso pode levar alguns minutos)")
        api.dataset_download_files(
            'phucthaiv02/butterfly-image-classification',
            path=str(temp_dir),
            unzip=True
        )
        
        print("✓ Download concluído!")
        return True
        
    except Exception as e:
        print(f"✗ Erro ao baixar: {e}")
        return False

def download_with_opendatasets():
    """Método alternativo usando opendatasets"""
    try:
        import opendatasets as od
        
        print("\n=== Baixando dataset com OpenDatasets ===")
        
        # Verifica se credenciais estão configuradas
        username = os.environ.get('KAGGLE_USERNAME')
        key = os.environ.get('KAGGLE_KEY')
        
        if not username or not key:
            print("✗ Credenciais não configuradas!")
            return False
        
        print("⏳ Baixando... (isso pode levar alguns minutos)")
        
        od.download(
            "https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification",
            data_dir="./dataset_temp"
        )
        
        print("✓ Download concluído!")
        return True
        
    except Exception as e:
        print(f"✗ Erro ao baixar: {e}")
        return False

def organize_dataset():
    """Organiza o dataset na estrutura correta"""
    print("\n=== Organizando estrutura de diretórios ===")
    
    # Procura pelo diretório baixado
    temp_path = Path('./dataset_temp')
    
    # Possíveis localizações do dataset
    possible_roots = [
        temp_path / 'butterfly-image-classification',
        temp_path,
    ]
    
    # Adiciona todos os subdiretórios encontrados
    if temp_path.exists():
        for item in temp_path.rglob('*'):
            if item.is_dir():
                possible_roots.append(item)
    
    # Encontra o diretório que contém train/test/val
    dataset_root = None
    for root in possible_roots:
        if root.exists():
            subdirs = [d.name for d in root.iterdir() if d.is_dir()]
            if 'train' in subdirs or 'test' in subdirs:
                dataset_root = root
                break
    
    if not dataset_root:
        print("✗ Não foi possível encontrar a estrutura do dataset!")
        print("Estrutura encontrada em dataset_temp:")
        if temp_path.exists():
            for item in temp_path.rglob('*'):
                if item.is_dir():
                    print(f"  - {item.relative_to(temp_path)}")
        return False
    
    print(f"✓ Dataset encontrado em: {dataset_root}")
    
    # Cria estrutura final
    final_dataset = Path('./dataset')
    final_dataset.mkdir(exist_ok=True)
    
    # Move diretórios
    splits_moved = []
    for split in ['train', 'test', 'valid', 'val']:
        src = dataset_root / split
        if src.exists():
            dst = final_dataset / ('val' if split == 'valid' else split)
            
            # Remove destino se já existir
            if dst.exists():
                shutil.rmtree(dst)
            
            shutil.copytree(src, dst)
            splits_moved.append(split)
            print(f"✓ Copiado: {split}/ -> dataset/{dst.name}/")
    
    if not splits_moved:
        print("✗ Nenhum diretório train/test/val encontrado!")
        return False
    
    # Remove temporários
    if temp_path.exists():
        shutil.rmtree(temp_path)
        print("✓ Arquivos temporários removidos")
    
    # Mostra estatísticas
    print("\n=== Estrutura final ===")
    total_images = 0
    total_species = 0
    
    for split_dir in sorted(final_dataset.iterdir()):
        if split_dir.is_dir():
            species_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            n_species = len(species_dirs)
            n_images = sum(
                len(list(d.glob('*.jpg'))) + 
                len(list(d.glob('*.jpeg'))) + 
                len(list(d.glob('*.png'))) + 
                len(list(d.glob('*.JPG')))
                for d in species_dirs
            )
            print(f"  📁 {split_dir.name}/")
            print(f"     └─ {n_species} espécies, {n_images} imagens")
            
            total_species = max(total_species, n_species)
            total_images += n_images
    
    print(f"\n📊 Total: {total_images} imagens de {total_species} espécies")
    
    return True

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    print("=== Verificando dependências Python ===")
    
    dependencies = {
        'kaggle': 'pip install kaggle',
        'opendatasets': 'pip install opendatasets',
        'dotenv': 'pip install python-dotenv'
    }
    
    missing = []
    for package, install_cmd in dependencies.items():
        try:
            if package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Execute: {install_cmd}")
            missing.append(package)
    
    if missing:
        print("\n⚠ Instalando dependências faltantes...")
        for package in missing:
            pkg_name = 'python-dotenv' if package == 'dotenv' else package
            os.system(f"pip install -q {pkg_name}")
        print("✓ Dependências instaladas!")
    
    return True

def main():
    print("=" * 70)
    print("  🦋 DOWNLOAD AUTOMÁTICO - BUTTERFLY IMAGE CLASSIFICATION")
    print("=" * 70)
    print()
    
    # Verifica e instala dependências
    check_dependencies()
    print()
    
    # Configura credenciais (suporta .env, variáveis de ambiente, ou interativo)
    if not setup_kaggle_credentials():
        sys.exit(1)
    
    print()
    
    # Tenta baixar
    success = False
    
    # Método 1: Kaggle API
    success = download_with_kaggle_api()
    
    # Método 2: OpenDatasets (fallback)
    if not success:
        print("\nTentando método alternativo...")
        success = download_with_opendatasets()
    
    if not success:
        print("\n✗ Falha no download!")
        print("\n💡 Alternativa: Download manual")
        print("1. Acesse: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification")
        print("2. Clique em 'Download'")
        print("3. Extraia o arquivo zip na pasta './dataset'")
        sys.exit(1)
    
    # Organiza estrutura
    if not organize_dataset():
        print("\n✗ Erro ao organizar dataset!")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("  ✅ DATASET PRONTO PARA USO!")
    print("=" * 70)
    print(f"\n📂 Localização: {Path('./dataset').absolute()}")
    print("\n🚀 Próximos passos:")
    print("   1. Compile o código C++:")
    print("      mkdir build && cd build")
    print("      cmake .. && make")
    print("      cd ..")
    print("\n   2. Execute o pré-processamento:")
    print("      ./build/preprocess_butterflies dataset/train preprocessed/train")

if __name__ == "__main__":
    main()