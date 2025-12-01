#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <map>
#include <cmath>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

Mat preprocessImage(const Mat& input) {
    Mat gray;
    if (input.channels() == 3) {
        cvtColor(input, gray, COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    Mat resized;
    resize(gray, resized, Size(224, 224), 0, 0, INTER_LINEAR);
    
    Mat equalized;
    equalizeHist(resized, equalized);
    
    Mat blended;
    addWeighted(equalized, 0.7, resized, 0.3, 0, blended);
    
    return blended;
}

vector<float> extractHOGFeaturesCompact(const Mat& img) {
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
    
    // Redução de dimensionalidade: média em blocos de 4
    vector<float> compact;
    compact.reserve(descriptors.size() / 4);
    for(size_t i = 0; i < descriptors.size(); i += 4) {
        float sum = 0;
        for(int j = 0; j < 4 && i+j < descriptors.size(); j++) {
            sum += descriptors[i+j];
        }
        compact.push_back(sum / 4.0f);
    }
    
    return compact;
}

vector<float> extractLBPFeaturesCompact(const Mat& img) {
    Mat resized;
    resize(img, resized, Size(64, 64), 0, 0, INTER_LINEAR);
    
    Mat lbp = Mat::zeros(resized.size(), CV_8UC1);
    
    for(int i = 1; i < resized.rows - 1; i++) {
        for(int j = 1; j < resized.cols - 1; j++) {
            uchar center = resized.at<uchar>(i, j);
            uchar code = 0;
            
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
    
    // Histograma com bins reduzidos (128 ao invés de 256)
    int histSize = 128;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;
    calcHist(&lbp, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    float sum = 0;
    for(int i = 0; i < histSize; i++) {
        sum += hist.at<float>(i);
    }
    
    vector<float> features;
    features.reserve(histSize);
    for(int i = 0; i < histSize; i++) {
        features.push_back(hist.at<float>(i) / (sum + 1e-7));
    }
    
    return features;
}

vector<float> extractColorHistogramCompact(const Mat& img_bgr) {
    Mat hsv;
    cvtColor(img_bgr, hsv, COLOR_BGR2HSV);
    
    // Bins reduzidos: 8x4x4 = 128 features (vs 16x8x8 = 1024)
    int histSize[] = {8, 4, 4};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    float vRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges, vRanges};
    int channels[] = {0, 1, 2};
    
    Mat hist;
    calcHist(&hsv, 1, channels, Mat(), hist, 3, histSize, ranges);
    normalize(hist, hist, 1, 0, NORM_L1);
    
    vector<float> features;
    features.reserve(128);
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
    Mat gray = preprocessImage(img_bgr);
    
    vector<float> hog = extractHOGFeaturesCompact(gray);
    vector<float> lbp = extractLBPFeaturesCompact(gray);
    vector<float> color = extractColorHistogramCompact(img_bgr);
    
    vector<float> combined;
    combined.reserve(hog.size() + lbp.size() + color.size());
    
    combined.insert(combined.end(), hog.begin(), hog.end());
    combined.insert(combined.end(), lbp.begin(), lbp.end());
    combined.insert(combined.end(), color.begin(), color.end());
    
    return combined;
}

vector<float> normalizeFeatureVector(const vector<float>& features) {
    double norm = 0.0;
    for(float f : features) {
        norm += f * f;
    }
    norm = sqrt(norm);
    
    vector<float> normalized;
    normalized.reserve(features.size());
    if(norm > 1e-7) {
        for(float f : features) {
            normalized.push_back(f / norm);
        }
    } else {
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
        
        line.erase(remove(line.begin(), line.end(), '\r'), line.end());
        stringstream ss(line);
        string filename, label;
        
        if (getline(ss, filename, ',') && getline(ss, label)) {
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
    auto start_time = chrono::high_resolution_clock::now();
    
    cout << "==============================================\n";
    cout << "BUTTERFLY FEATURE EXTRACTION\n";
    cout << "==============================================\n";
    
    string trainCSV = "dataset/Training_set.csv";
    string trainDir = "dataset/train";
    string outputFile = "features_combined.csv";
    
    cout << "\n[1/3] Carregando labels...\n";
    map<string, string> trainLabels = loadLabelsFromCSV(trainCSV);
    
    if (trainLabels.empty()) {
        cerr << "ERRO: Nenhum label carregado!\n";
        return -1;
    }
    
    cout << "  Labels: " << trainLabels.size() << endl;
    
    map<string, int> speciesCount;
    for (const auto& [filename, label] : trainLabels) {
        speciesCount[label]++;
    }
    cout << "  Especies: " << speciesCount.size() << endl;
    
    if (!fs::exists(trainDir)) {
        cerr << "\nERRO: Diretorio nao encontrado: " << trainDir << endl;
        return -1;
    }
    
    // Coletar todas as imagens
    vector<pair<string, string>> image_label_pairs;
    for (const auto& entry : fs::directory_iterator(trainDir)) {
        if (!entry.is_regular_file() || !isImageFile(entry.path().string())) {
            continue;
        }
        
        string filename = entry.path().filename().string();
        auto it = trainLabels.find(filename);
        if (it != trainLabels.end()) {
            image_label_pairs.push_back({entry.path().string(), it->second});
        }
    }
    
    cout << "\n[2/3] Extraindo features (paralelizado)...\n";
    cout << "  Imagens a processar: " << image_label_pairs.size() << endl;
    
    int total = image_label_pairs.size();
    vector<string> results(total);
    int processed = 0;
    int errors = 0;
    
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    cout << "  Threads OpenMP: " << num_threads << endl;
    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 10) reduction(+:processed,errors)
#endif
    for(int idx = 0; idx < total; idx++) {
        try {
            const auto& [path, label] = image_label_pairs[idx];
            
            Mat img = imread(path, IMREAD_COLOR);
            if (img.empty()) {
                errors++;
                continue;
            }
            
            vector<float> rawFeatures = extractCombinedFeatures(img);
            vector<float> features = normalizeFeatureVector(rawFeatures);
            
            stringstream ss;
            ss << label;
            for(float f : features) {
                ss << "," << f;
            }
            ss << "\n";
            
            results[idx] = ss.str();
            processed++;
            
            if (processed % 200 == 0) {
#pragma omp critical
                {
                    cout << "    Progresso: " << processed << "/" << total << "\r" << flush;
                }
            }
            
        } catch (const exception& e) {
            errors++;
        }
    }
    
    cout << "\n  Processadas: " << processed << " imagens\n";
    
    // Escrever arquivo
    cout << "\n[3/3] Salvando features...\n";
    ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        cerr << "ERRO: Nao conseguiu criar " << outputFile << endl;
        return -1;
    }
    
    for(const auto& line : results) {
        if(!line.empty()) {
            outFile << line;
        }
    }
    outFile.close();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    
    cout << "\n==============================================\n";
    cout << "CONCLUIDO!\n";
    cout << "==============================================\n";
    cout << "  Processadas: " << processed << " imagens\n";
    cout << "  Erros: " << errors << endl;
    cout << "  Tempo total: " << duration.count() << "s\n";
    cout << "  Arquivo: " << outputFile << endl;
    cout << "==============================================\n";
    
    return 0;
}