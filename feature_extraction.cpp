#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <map>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

/**
 * Extração de Features HOG (Histogram of Oriented Gradients)
 * 
 * Por que HOG?
 * - Captura bordas e formas (padrões geométricos das asas)
 * - Robusto a variações de iluminação
 * - Padrão para classificação de objetos
 * 
 * Parâmetros escolhidos:
 * - winSize: 64x128 (janela padrão)
 * - blockSize: 16x16 (normalização local)
 * - blockStride: 8x8 (overlap de 50%)
 * - cellSize: 8x8 (resolução fina para detalhes)
 * - nbins: 9 (direções de gradiente)
 */
vector<float> extractHOGFeatures(Mat img) {
    Size winSize(64, 128); 
    Mat resizedImg;
    resize(img, resizedImg, winSize);

    HOGDescriptor hog(
        winSize,      // Tamanho da janela
        Size(16, 16), // Tamanho do bloco
        Size(8, 8),   // Stride do bloco
        Size(8, 8),   // Tamanho da célula
        9             // Número de bins (direções)
    );

    vector<float> descriptors;
    hog.compute(resizedImg, descriptors);
    return descriptors;
}

/**
 * Extração de Features LBP (Local Binary Patterns)
 * 
 * Por que LBP?
 * - Captura texturas locais (micro-padrões)
 * - Complementa HOG (textura vs forma)
 * - Invariante a transformações monotônicas de intensidade
 * - Muito eficiente computacionalmente
 * 
 * Implementação:
 * - LBP 3x3 básico
 * - Compara vizinhos 8-conectados com pixel central
 * - Gera código binário de 8 bits
 * - Histograma de 256 bins como feature vector
 */
vector<float> extractLBPFeatures(Mat img) {
    Mat resizedImg;
    resize(img, resizedImg, Size(128, 128));
    
    Mat lbp = Mat::zeros(resizedImg.size(), CV_8UC1);
    
    // LBP básico 3x3
    for(int i = 1; i < resizedImg.rows - 1; i++) {
        for(int j = 1; j < resizedImg.cols - 1; j++) {
            uchar center = resizedImg.at<uchar>(i, j);
            uchar code = 0;
            
            // Compara 8 vizinhos com o centro (sentido horário)
            code |= (resizedImg.at<uchar>(i-1, j-1) >= center) << 7;
            code |= (resizedImg.at<uchar>(i-1, j)   >= center) << 6;
            code |= (resizedImg.at<uchar>(i-1, j+1) >= center) << 5;
            code |= (resizedImg.at<uchar>(i, j+1)   >= center) << 4;
            code |= (resizedImg.at<uchar>(i+1, j+1) >= center) << 3;
            code |= (resizedImg.at<uchar>(i+1, j)   >= center) << 2;
            code |= (resizedImg.at<uchar>(i+1, j-1) >= center) << 1;
            code |= (resizedImg.at<uchar>(i, j-1)   >= center) << 0;
            
            lbp.at<uchar>(i, j) = code;
        }
    }
    
    // Calcular histograma (256 bins)
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;
    calcHist(&lbp, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    // Converter para vector
    vector<float> features;
    for(int i = 0; i < histSize; i++) {
        features.push_back(hist.at<float>(i));
    }
    
    return features;
}

/**
 * Features Combinadas (HOG + LBP)
 * 
 * Por que combinar?
 * - HOG captura FORMA (bordas, contornos)
 * - LBP captura TEXTURA (padrões locais)
 * - Complementares: melhor performance que isolados
 * 
 * Dimensão final: ~3.256 features
 * - HOG: ~3.000 features
 * - LBP: 256 features (histograma)
 */
vector<float> extractCombinedFeatures(Mat img) {
    vector<float> hog = extractHOGFeatures(img);
    vector<float> lbp = extractLBPFeatures(img);
    
    // Concatenar HOG e LBP
    hog.insert(hog.end(), lbp.begin(), lbp.end());
    return hog;
}

/**
 * Normalização L2
 * 
 * Por que normalizar?
 * - Equaliza escalas diferentes entre features
 * - Melhora convergência do SVM
 * - Reduz impacto de outliers
 */
vector<float> normalizeFeatures(vector<float>& features) {
    vector<float> normalized;
    normalize(features, normalized, 1.0, 0.0, NORM_L2);
    return normalized;
}

/**
 * Carrega mapeamento filename -> label do CSV
 * 
 * IMPORTANTE: Adaptado para estrutura flat do dataset
 * - dataset/train/ contém TODAS as imagens
 * - Training_set.csv mapeia filename para label
 * 
 * Formato CSV:
 * filename,label
 * Image_1.jpg,SOUTHERN DOGFACE
 * Image_2.jpg,ADONIS
 * ...
 */
map<string, string> loadLabelsFromCSV(const string& csvPath) {
    map<string, string> labels;
    
    ifstream file(csvPath);
    if (!file.is_open()) {
        cerr << "ERRO: Nao foi possivel abrir " << csvPath << endl;
        return labels;
    }
    
    string line;
    bool firstLine = true;
    
    while (getline(file, line)) {
        // Pular header
        if (firstLine) {
            firstLine = false;
            continue;
        }
        
        // Parse CSV: filename,label
        stringstream ss(line);
        string filename, label;
        
        if (getline(ss, filename, ',') && getline(ss, label)) {
            // Remover espaços e \r\n (compatibilidade Windows)
            filename.erase(remove(filename.begin(), filename.end(), '\r'), filename.end());
            label.erase(remove(label.begin(), label.end(), '\r'), label.end());
            
            labels[filename] = label;
        }
    }
    
    file.close();
    return labels;
}

/**
 * Verifica se arquivo é uma imagem válida
 */
bool isImageFile(string path) {
    string ext = fs::path(path).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

int main() {
    // ========== CONFIGURAÇÕES ==========
    string trainCSV = "dataset/Training_set.csv";
    string trainDir = "dataset/train";
    string testCSV = "dataset/Testing_set.csv";
    string testDir = "dataset/test";
    
    // ESCOLHA O TIPO DE FEATURE:
    // 1 = HOG apenas (para ablation study)
    // 2 = LBP apenas (para ablation study)
    // 3 = HOG + LBP combinados (baseline)
    int featureType = 3;  // <--- MUDE AQUI para gerar diferentes CSVs!
    
    string outputFile;
    if (featureType == 1) {
        outputFile = "features_hog.csv";
        cout << "Modo: HOG" << endl;
    } else if (featureType == 2) {
        outputFile = "features_lbp.csv";
        cout << "Modo: LBP" << endl;
    } else {
        outputFile = "features_combined.csv";
        cout << "Modo: HOG + LBP Combinados" << endl;
    }
    
    cout << "================================================" << endl;
    
    // ====================================
    
    // 1. CARREGAR LABELS DO CSV
    cout << "Carregando labels do CSV..." << endl;
    map<string, string> trainLabels = loadLabelsFromCSV(trainCSV);
    
    if (trainLabels.empty()) {
        cerr << "ERRO: Nenhum label carregado de " << trainCSV << endl;
        cerr << "Verifique se o arquivo existe e tem o formato correto." << endl;
        return -1;
    }
    
    cout << "✓ Labels carregados: " << trainLabels.size() << " amostras" << endl;
    
    // Contar espécies únicas
    map<string, int> speciesCount;
    for (const auto& pair : trainLabels) {
        speciesCount[pair.second]++;
    }
    cout << "✓ Especies unicas: " << speciesCount.size() << endl;
    
    // Mostrar algumas espécies
    cout << "\nPrimeiras 5 especies:" << endl;
    int shown = 0;
    for (const auto& pair : speciesCount) {
        if (shown++ >= 5) break;
        cout << "  - " << pair.first << ": " << pair.second << " imagens" << endl;
    }
    
    cout << "\n================================================" << endl;
    cout << "Iniciando extracao de features..." << endl;
    cout << "================================================" << endl;

    // 2. ABRIR ARQUIVO DE SAÍDA
    ofstream file(outputFile);
    if (!file.is_open()) {
        cerr << "Erro ao criar arquivo " << outputFile << endl;
        return -1;
    }
    file.close();

    int count = 0;
    int notFound = 0;
    int featureDim = 0;

    // 3. PROCESSAR IMAGENS DO TRAIN
    if (!fs::exists(trainDir)) {
        cerr << "ERRO: Diretorio " << trainDir << " nao encontrado!" << endl;
        return -1;
    }
    
    for (const auto& entry : fs::directory_iterator(trainDir)) {
        if (!entry.is_regular_file()) continue;
        if (!isImageFile(entry.path().string())) continue;
        
        string filename = entry.path().filename().string();
        
        // Buscar label no mapa
        auto it = trainLabels.find(filename);
        if (it == trainLabels.end()) {
            // Imagem sem label no CSV, pular
            notFound++;
            continue;
        }
        
        string label = it->second;
        
        // Carregar imagem em GRAYSCALE
        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "AVISO: Nao foi possivel carregar " << filename << endl;
            continue;
        }
        
        // Extrair features conforme featureType
        vector<float> rawFeatures;
        if (featureType == 1) {
            rawFeatures = extractHOGFeatures(img);
        } else if (featureType == 2) {
            rawFeatures = extractLBPFeatures(img);
        } else {
            rawFeatures = extractCombinedFeatures(img);
        }
        
        // Normalizar
        vector<float> finalFeatures = normalizeFeatures(rawFeatures);
        
        // Armazenar dimensão das features (primeira vez)
        if (count == 0) {
            featureDim = finalFeatures.size();
        }
        
        // Salvar no CSV (formato: label,feature1,feature2,...)
        ofstream outFile(outputFile, ios_base::app);
        if (outFile.is_open()) {
            outFile << label;
            for (float f : finalFeatures) {
                outFile << "," << f;
            }
            outFile << "\n";
            outFile.close();
            
            count++;
            if (count % 100 == 0) {
                cout << "Processadas " << count << " imagens..." << endl;
            }
        }
    }

    // ========== ESTATÍSTICAS FINAIS ==========
    cout << "================================================" << endl;
    
    if (count == 0) {
        cerr << "ERRO: Nenhuma imagem foi processada!" << endl;
        return -1;
    }
    
    cout << "SUCESSO! Processamento concluido." << endl;
    cout << "------------------------------------------------" << endl;
    cout << "Total de imagens processadas: " << count << endl;
    cout << "Imagens sem label no CSV: " << notFound << endl;
    cout << "Classes/especies: " << speciesCount.size() << endl;
    cout << "Dimensao do vetor de features: " << featureDim << endl;
    cout << "Tipo de feature extraida: ";
    if (featureType == 1) cout << "HOG" << endl;
    else if (featureType == 2) cout << "LBP" << endl;
    else cout << "HOG + LBP" << endl;
    cout << "Arquivo de saida: " << outputFile << endl;
    cout << "================================================" << endl;

    return 0;
}