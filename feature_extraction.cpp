#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// --- 1. Extração HOG ---
vector<float> extractHOGFeatures(Mat img) {
    Size winSize(64, 128); 
    Mat resizedImg;
    resize(img, resizedImg, winSize);

    HOGDescriptor hog(
        winSize, 
        Size(16, 16), 
        Size(8, 8),   
        Size(8, 8),   
        9             
    );

    vector<float> descriptors;
    hog.compute(resizedImg, descriptors);
    return descriptors;
}

// --- 2. Extração LBP ---
vector<float> extractLBPFeatures(Mat img) {
    Mat resizedImg;
    resize(img, resizedImg, Size(128, 128));
    
    Mat lbp = Mat::zeros(resizedImg.size(), CV_8UC1);
    
    // LBP básico 3x3
    for(int i = 1; i < resizedImg.rows - 1; i++) {
        for(int j = 1; j < resizedImg.cols - 1; j++) {
            uchar center = resizedImg.at<uchar>(i, j);
            uchar code = 0;
            
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
    
    // Calcular histograma
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;
    calcHist(&lbp, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    vector<float> features;
    for(int i = 0; i < histSize; i++) {
        features.push_back(hist.at<float>(i));
    }
    
    return features;
}

// --- 3. Features Combinadas (HOG + LBP) ---
vector<float> extractCombinedFeatures(Mat img) {
    vector<float> hog = extractHOGFeatures(img);
    vector<float> lbp = extractLBPFeatures(img);
    
    // Concatenar HOG e LBP
    hog.insert(hog.end(), lbp.begin(), lbp.end());
    return hog;
}

// --- 4. Normalização ---
vector<float> normalizeFeatures(vector<float>& features) {
    vector<float> normalized;
    normalize(features, normalized, 1.0, 0.0, NORM_L2);
    return normalized;
}

// --- 5. Função Auxiliar para verificar extensão ---
bool isImageFile(string path) {
    string ext = fs::path(path).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

int main() {
    // ========== CONFIGURAÇÕES ==========
    string datasetPath = "dataset"; 
    
    // ESCOLHA O TIPO DE FEATURE:
    // 1 = HOG apenas
    // 2 = LBP apenas
    // 3 = HOG + LBP combinados
    int featureType = 3;  // <--- MUDE AQUI!
    
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
    
    // ====================================

    // Abre o arquivo CSV
    ofstream file(outputFile);
    if (!file.is_open()) {
        cerr << "Erro ao criar arquivo " << outputFile << endl;
        return -1;
    }
    file.close();

    int count = 0;
    int featureDim = 0;
    cout << "Iniciando processamento do dataset em: " << datasetPath << endl;
    cout << "================================================" << endl;

    // --- LOOP RECURSIVO ---
    if (fs::exists(datasetPath)) {
        for (const auto& entry : fs::recursive_directory_iterator(datasetPath)) {
            if (entry.is_regular_file()) {
                string filePath = entry.path().string();
                
                if (isImageFile(filePath)) {
                    // 1. Descobrir o Label
                    string label = entry.path().parent_path().filename().string();

                    // 2. Carregar Imagem
                    Mat img = imread(filePath, IMREAD_GRAYSCALE);
                    if (img.empty()) continue;

                    // 3. Extrair Features (de acordo com o tipo escolhido)
                    vector<float> rawFeatures;
                    if (featureType == 1) {
                        rawFeatures = extractHOGFeatures(img);
                    } else if (featureType == 2) {
                        rawFeatures = extractLBPFeatures(img);
                    } else {
                        rawFeatures = extractCombinedFeatures(img);
                    }
                    
                    // 4. Normalizar
                    vector<float> finalFeatures = normalizeFeatures(rawFeatures);
                    
                    // Armazenar dimensão das features (primeira vez)
                    if (count == 0) {
                        featureDim = finalFeatures.size();
                    }

                    // 5. Salvar no CSV
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
                            cout << "Processadas " << count << " imagens... (Ultima: " << label << ")" << endl;
                        }
                    }
                }
            }
        }
    } else {
        cerr << "Pasta 'dataset' nao encontrada! Verifique o caminho." << endl;
        return -1;
    }

    // ========== ESTATÍSTICAS FINAIS ==========
    cout << "================================================" << endl;
    cout << "SUCESSO! Processamento concluido." << endl;
    cout << "------------------------------------------------" << endl;
    cout << "Total de imagens processadas: " << count << endl;
    cout << "Dimensao do vetor de features: " << featureDim << endl;
    cout << "Tipo de feature extraida: ";
    if (featureType == 1) cout << "HOG" << endl;
    else if (featureType == 2) cout << "LBP" << endl;
    else cout << "HOG + LBP" << endl;
    cout << "Arquivo de saida: " << outputFile << endl;
    cout << "================================================" << endl;

    return 0;
}