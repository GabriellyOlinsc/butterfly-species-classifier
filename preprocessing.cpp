#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class ImagePreprocessor {
private:
    // Calcula métrica simples de qualidade
    double calculateQuality(const cv::Mat& img1, const cv::Mat& img2) {
        cv::Mat diff;
        cv::absdiff(img1, img2, diff);
        cv::Scalar sum = cv::sum(diff);
        double mse = (sum[0] + sum[1] + sum[2]) / (img1.total() * img1.channels());
        return 10.0 * log10(255.0 * 255.0 / (mse + 1e-10)); // PSNR simplificado
    }

public:
    // Pipeline de pré-processamento simplificado
    cv::Mat preprocess(const cv::Mat& input, bool verbose = false) {
        if (verbose) {
            std::cout << "Processando imagem " << input.cols << "x" << input.rows << std::endl;
        }
        
        cv::Mat current = input.clone();
        
        // 1. Redimensionar para 224x224 (padrão para classificação)
        cv::resize(current, current, cv::Size(224, 224), 0, 0, cv::INTER_LANCZOS4);
        
        // 2. Remoção de ruído (bilateral filter preserva bordas)
        cv::Mat denoised;
        cv::bilateralFilter(current, denoised, 9, 75, 75);
        
        // 3. Equalização adaptativa (CLAHE) para melhorar contraste
        cv::Mat lab;
        cv::cvtColor(denoised, lab, cv::COLOR_BGR2Lab);
        
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(channels[0], channels[0]);
        
        cv::merge(channels, lab);
        cv::Mat result;
        cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
        
        // 4. Sharpening leve (realça detalhes)
        cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
        cv::Mat sharpened;
        cv::filter2D(result, sharpened, -1, kernel);
        
        // Blend: 70% equalizado + 30% sharpened
        cv::addWeighted(result, 0.7, sharpened, 0.3, 0, result);
        
        if (verbose) {
            double quality = calculateQuality(input, result);
            std::cout << "  PSNR: " << quality << " dB" << std::endl;
        }
        
        return result;
    }
    
    // Processa diretório FLAT (sem subpastas)
    int batchProcessFlat(const std::string& input_dir, const std::string& output_dir) {
        fs::create_directories(output_dir);
        
        int count = 0;
        std::cout << "\n=== Processando imagens de " << input_dir << " ===" << std::endl;
        
        if (!fs::exists(input_dir)) {
            std::cerr << "ERRO: Diretorio " << input_dir << " nao existe!" << std::endl;
            return 0;
        }
        
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (!entry.is_regular_file()) continue;
            
            std::string ext = entry.path().extension().string();
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".JPG") continue;
            
            std::string filename = entry.path().filename().string();
            fs::path out_path = fs::path(output_dir) / filename;
            
            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) {
                std::cout << "  ERRO ao carregar: " << filename << std::endl;
                continue;
            }
            
            cv::Mat processed = preprocess(img);
            cv::imwrite(out_path.string(), processed);
            
            count++;
            
            if (count % 50 == 0) {
                std::cout << "[Progresso: " << count << " imagens]" << std::endl;
            }
        }
        
        return count;
    }
};

int main(int argc, char** argv) {
    std::cout << "=== Butterfly Image Preprocessor (Flat Structure) ===" << std::endl;
    
    if (argc < 3) {
        std::cout << "\nUso: " << argv[0] << " <input_dir> <output_dir>" << std::endl;
        std::cout << "\nExemplo:" << std::endl;
        std::cout << "  " << argv[0] << " dataset/train preprocessed/train" << std::endl;
        std::cout << "\nOBS: Funciona com estrutura FLAT (todas as imagens na mesma pasta)" << std::endl;
        return 1;
    }
    
    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    
    if (!fs::exists(input_dir)) {
        std::cerr << "\nERRO: Diretório não encontrado: " << input_dir << std::endl;
        return 1;
    }
    
    ImagePreprocessor preprocessor;
    int total = preprocessor.batchProcessFlat(input_dir, output_dir);
    
    std::cout << "\n=== Concluído ===" << std::endl;
    std::cout << "Total processado: " << total << " imagens" << std::endl;
    std::cout << "Salvo em: " << output_dir << std::endl;
    
    return 0;
}