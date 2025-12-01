#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <map>
#include <cmath>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Pré-processamento simples e reproduzível
Mat preprocessImage(const Mat& input) {
    Mat gray;
    
    // Garantir que é grayscale
    if (input.channels() == 3) {
        cvtColor(input, gray, COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Resize para tamanho fixo
    Mat resized;
    resize(gray, resized, Size(224, 224), 0, 0, INTER_LINEAR);
    
    // Equalização suave (melhora contraste sem distorcer)
    Mat equalized;
    equalizeHist(resized, equalized);
    
    // Blend: 70% equalizado + 30% original (evita over-equalização)
    Mat blended;
    addWeighted(equalized, 0.7, resized, 0.3, 0, blended);
    
    return blended;
}

vector<float> extractHOGFeatures(const Mat& img) {
    // Redimensionar para tamanho HOG
    Mat resized;
    resize(img, resized, Size(64, 128), 0, 0, INTER_LINEAR);

    HOGDescriptor hog(
        Size(64, 128),
        Size(16, 16),
        Size(8, 8),
        Size(8, 8),
        9
    );

    vector<float> descriptors;
    hog.compute(resized, descriptors);
    
    return descriptors;
}

vector<float> extractLBPFeatures(const Mat& img) {
    // Redimensionar
    Mat resized;
    resize(img, resized, Size(128, 128), 0, 0, INTER_LINEAR);
    
    Mat lbp = Mat::zeros(resized.size(), CV_8UC1);
    
    // LBP 3x3
    for(int i = 1; i < resized.rows - 1; i++) {
        for(int j = 1; j < resized.cols - 1; j++) {
            uchar center = resized.at<uchar>(i, j);
            uchar code = 0;
            
            // Ordem horária dos 8 vizinhos
            code |= (resized.at<uchar>(i-1, j-1) >= center) << 7;
            code |= (resized.at<uchar>(i-1, j)   >= center) << 6;
            code |= (resized.at<uchar>(i-1, j+1) >= center) << 5;
            code |= (resized.at<uchar>(i, j+1)   >= center) << 4;
            code |= (resized.at<uchar>(i+1, j+1) >= center) << 3;
            code |= (resized.at<uchar>(i+1, j)   >= center) << 2;
            code |= (resized.at<uchar>(i+1, j-1) >= center) << 1;
            code |= (resized.at<uchar>(i, j-1)   >= center) << 0;
            
            lbp.at<uchar>(i, j) = code;
        }
    }
    
    // Histograma normalizado
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;
    calcHist(&lbp, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    // Normalizar histograma (soma = 1)
    float sum = 0;
    for(int i = 0; i < histSize; i++) {
        sum += hist.at<float>(i);
    }
    
    vector<float> features;
    for(int i = 0; i < histSize; i++) {
        features.push_back(hist.at<float>(i) / (sum + 1e-7));
    }
    
    return features;
}

vector<float> extractColorHistogram(const Mat& img_bgr) {
    /**
     * NOVO: Color Histogram (complementa HOG+LBP)
     * Captura informações de cor que HOG/LBP perdem
     */
    Mat hsv;
    cvtColor(img_bgr, hsv, COLOR_BGR2HSV);
    
    // Histograma 3D: Hue (16 bins), Saturation (8 bins), Value (8 bins)
    int histSize[] = {16, 8, 8};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    float vRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges, vRanges};
    int channels[] = {0, 1, 2};
    
    Mat hist;
    calcHist(&hsv, 1, channels, Mat(), hist, 3, histSize, ranges);
    
    // Normalizar
    normalize(hist, hist, 1, 0, NORM_L1);
    
    // Converter para vector
    vector<float> features;
    for(int h = 0; h < histSize[0]; h++) {
        for(int s = 0; s < histSize[1]; s++) {
            for(int v = 0; v < histSize[2]; v++) {
                features.push_back(hist.at<float>(h, s, v));
            }
        }
    }
    
    return features;
}

vector<float> extractCombinedFeatures(const Mat& img_bgr) {
    /**
     * Features combinadas: HOG + LBP + Color
     * 
     * - HOG: forma/bordas (~3780 dims)
     * - LBP: textura (~256 dims)
     * - Color: aparência (~1024 dims)
     * 
     * Total: ~5060 features
     */
    
    // Processar imagem
    Mat gray = preprocessImage(img_bgr);
    
    // Extrair features
    vector<float> hog = extractHOGFeatures(gray);
    vector<float> lbp = extractLBPFeatures(gray);
    vector<float> color = extractColorHistogram(img_bgr);  // usa imagem colorida
    
    // Combinar (concatenar)
    vector<float> combined;
    combined.reserve(hog.size() + lbp.size() + color.size());
    
    combined.insert(combined.end(), hog.begin(), hog.end());
    combined.insert(combined.end(), lbp.begin(), lbp.end());
    combined.insert(combined.end(), color.begin(), color.end());
    
    return combined;
}

vector<float> normalizeFeatureVector(const vector<float>& features) {
    /**
     * NORMALIZAÇÃO CORRIGIDA
     * 
     * Usa L2 normalization corretamente:
     * normalized[i] = features[i] / ||features||_2
     */
    double norm = 0.0;
    for(float f : features) {
        norm += f * f;
    }
    norm = sqrt(norm);
    
    vector<float> normalized;
    if(norm > 1e-7) {
        for(float f : features) {
            normalized.push_back(f / norm);
        }
    } else {
        // Se norm é zero, retorna zeros
        normalized = features;
    }
    
    return normalized;
}

map<string, string> loadLabelsFromCSV(const string& csvPath) {
    map<string, string> labels;
    
    ifstream file(csvPath);
    if (!file.is_open()) {
        cerr << "ERRO: Nao conseguiu abrir " << csvPath << endl;
        return labels;
    }
    
    string line;
    bool firstLine = true;
    
    while (getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }
        
        // Remove \r (Windows line ending)
        line.erase(remove(line.begin(), line.end(), '\r'), line.end());
        
        stringstream ss(line);
        string filename, label;
        
        if (getline(ss, filename, ',') && getline(ss, label)) {
            // Trim whitespace
            label.erase(0, label.find_first_not_of(" \t\n\r"));
            label.erase(label.find_last_not_of(" \t\n\r") + 1);
            
            labels[filename] = label;
        }
    }
    
    file.close();
    return labels;
}

bool isImageFile(const string& path) {
    string ext = fs::path(path).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

int main() {
    cout << "================================================" << endl;
    cout << "BUTTERFLY FEATURE EXTRACTION V2" << endl;
    cout << "================================================" << endl;
    
    string trainCSV = "dataset/Training_set.csv";
    string trainDir = "dataset/train";
    string outputFile = "features_combined.csv";
    
    // Carregar labels
    cout << "\n[1/3] Carregando labels..." << endl;
    map<string, string> trainLabels = loadLabelsFromCSV(trainCSV);
    
    if (trainLabels.empty()) {
        cerr << "ERRO: Nenhum label carregado!" << endl;
        return -1;
    }
    
    cout << "  ✓ Labels: " << trainLabels.size() << endl;
    
    // Contar espécies
    map<string, int> speciesCount;
    for (const auto& [filename, label] : trainLabels) {
        speciesCount[label]++;
    }
    cout << "  ✓ Especies: " << speciesCount.size() << endl;
    
    // Mostrar distribuição
    cout << "\n  Top 5 especies mais frequentes:" << endl;
    vector<pair<string, int>> sortedSpecies(speciesCount.begin(), speciesCount.end());
    sort(sortedSpecies.begin(), sortedSpecies.end(), 
         [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for(int i = 0; i < min(5, (int)sortedSpecies.size()); i++) {
        cout << "    " << (i+1) << ". " << sortedSpecies[i].first 
             << ": " << sortedSpecies[i].second << " imagens" << endl;
    }
    
    // Verificar diretório
    if (!fs::exists(trainDir)) {
        cerr << "\nERRO: Diretorio nao encontrado: " << trainDir << endl;
        return -1;
    }
    
    // Abrir arquivo de saída
    ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        cerr << "ERRO: Nao conseguiu criar " << outputFile << endl;
        return -1;
    }
    
    cout << "\n[2/3] Extraindo features..." << endl;
    cout << "  (Isso pode demorar alguns minutos)" << endl;
    
    int processed = 0;
    int notFound = 0;
    int errors = 0;
    int featureDim = 0;
    
    // Processar todas as imagens
    for (const auto& entry : fs::directory_iterator(trainDir)) {
        if (!entry.is_regular_file() || !isImageFile(entry.path().string())) {
            continue;
        }
        
        string filename = entry.path().filename().string();
        
        // Buscar label
        auto it = trainLabels.find(filename);
        if (it == trainLabels.end()) {
            notFound++;
            continue;
        }
        
        string label = it->second;
        
        // Carregar imagem (BGR para ter cores)
        Mat img = imread(entry.path().string(), IMREAD_COLOR);
        if (img.empty()) {
            cerr << "  ERRO ao carregar: " << filename << endl;
            errors++;
            continue;
        }
        
        try {
            // Extrair features
            vector<float> rawFeatures = extractCombinedFeatures(img);
            
            // Normalizar
            vector<float> features = normalizeFeatureVector(rawFeatures);
            
            // Salvar dimensão (primeira iteração)
            if (processed == 0) {
                featureDim = features.size();
            }
            
            // Escrever no CSV: label,f1,f2,...,fn
            outFile << label;
            for(float f : features) {
                outFile << "," << f;
            }
            outFile << "\n";
            
            processed++;
            
            // Progress
            if (processed % 100 == 0) {
                cout << "    Processadas: " << processed << " imagens..." << endl;
            }
            
        } catch (const exception& e) {
            cerr << "  ERRO em " << filename << ": " << e.what() << endl;
            errors++;
        }
    }
    
    outFile.close();
    
    // Relatório final
    cout << "\n[3/3] Concluido!" << endl;
    cout << "================================================" << endl;
    cout << "ESTATISTICAS:" << endl;
    cout << "  ✓ Processadas: " << processed << " imagens" << endl;
    cout << "  ✗ Sem label: " << notFound << endl;
    cout << "  ✗ Erros: " << errors << endl;
    cout << "  • Especies: " << speciesCount.size() << endl;
    cout << "  • Dimensao features: " << featureDim << endl;
    cout << "  • Arquivo: " << outputFile << endl;
    cout << "================================================" << endl;
    
    if (processed == 0) {
        cerr << "\nERRO FATAL: Nenhuma imagem foi processada!" << endl;
        return -1;
    }
    
    cout << "\nProximo passo: python3 train_classifier.py" << endl;
    
    return 0;
}