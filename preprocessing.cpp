#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class ImagePreprocessor {
private:
    // Calcula PSNR (Peak Signal-to-Noise Ratio)
    double calculatePSNR(const cv::Mat& original, const cv::Mat& processed) {
        cv::Mat diff;
        cv::absdiff(original, processed, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);
        
        cv::Scalar s = cv::sum(diff);
        double sse = s.val[0] + s.val[1] + s.val[2];
        
        if (sse <= 1e-10) return 0.0;
        
        double mse = sse / (double)(original.channels() * original.total());
        double psnr = 10.0 * log10((255.0 * 255.0) / mse);
        
        return psnr;
    }
    
    // Calcula SSIM (Structural Similarity Index)
    double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {
        const double C1 = 6.5025, C2 = 58.5225;
        
        cv::Mat I1, I2;
        img1.convertTo(I1, CV_32F);
        img2.convertTo(I2, CV_32F);
        
        cv::Mat I1_2 = I1.mul(I1);
        cv::Mat I2_2 = I2.mul(I2);
        cv::Mat I1_I2 = I1.mul(I2);
        
        cv::Mat mu1, mu2;
        cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
        cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
        
        cv::Mat mu1_2 = mu1.mul(mu1);
        cv::Mat mu2_2 = mu2.mul(mu2);
        cv::Mat mu1_mu2 = mu1.mul(mu2);
        
        cv::Mat sigma1_2, sigma2_2, sigma12;
        cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;
        
        cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;
        
        cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;
        
        cv::Mat t1, t2, t3;
        t1 = 2 * mu1_mu2 + C1;
        t2 = 2 * sigma12 + C2;
        t3 = t1.mul(t2);
        
        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        t1 = t1.mul(t2);
        
        cv::Mat ssim_map;
        cv::divide(t3, t1, ssim_map);
        
        cv::Scalar mssim = cv::mean(ssim_map);
        return (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3.0;
    }

public:
    struct PreprocessingMetrics {
        double psnr_after_denoise;
        double ssim_after_denoise;
        double psnr_after_norm;
        double ssim_after_norm;
        double psnr_after_clahe;
        double ssim_after_clahe;
    };
    
    // Pipeline completo de pré-processamento
    cv::Mat preprocess(const cv::Mat& input, PreprocessingMetrics& metrics, 
                       bool save_intermediate = false, 
                       const std::string& output_prefix = "") {
        cv::Mat original = input.clone();
        cv::Mat current = input.clone();
        
        std::cout << "=== Iniciando Pré-processamento ===" << std::endl;
        std::cout << "Dimensões originais: " << input.cols << "x" << input.rows << std::endl;
        
        // 1. Redimensionamento para tamanho padrão (224x224 comum em classificação)
        cv::Mat resized;
        cv::resize(current, resized, cv::Size(224, 224), 0, 0, cv::INTER_LANCZOS4);
        current = resized.clone();
        std::cout << "Redimensionado para: 224x224" << std::endl;
        
        if (save_intermediate) {
            cv::imwrite(output_prefix + "_1_resized.jpg", current);
        }
        
        // 2. Remoção de ruído com bilateral filter (preserva bordas)
        cv::Mat denoised;
        cv::bilateralFilter(current, denoised, 9, 75, 75);
        
        metrics.psnr_after_denoise = calculatePSNR(current, denoised);
        metrics.ssim_after_denoise = calculateSSIM(current, denoised);
        
        std::cout << "\n--- Após Denoise (Bilateral Filter) ---" << std::endl;
        std::cout << "PSNR: " << metrics.psnr_after_denoise << " dB" << std::endl;
        std::cout << "SSIM: " << metrics.ssim_after_denoise << std::endl;
        
        current = denoised.clone();
        
        if (save_intermediate) {
            cv::imwrite(output_prefix + "_2_denoised.jpg", current);
        }
        
        // 3. Normalização de cor (conversão para Lab e normalização de luminosidade)
        cv::Mat lab;
        cv::cvtColor(current, lab, cv::COLOR_BGR2Lab);
        
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        
        // Normaliza canal L (luminosidade)
        cv::Mat l_normalized;
        cv::normalize(lab_channels[0], l_normalized, 0, 255, cv::NORM_MINMAX);
        lab_channels[0] = l_normalized;
        
        cv::Mat lab_normalized;
        cv::merge(lab_channels, lab_normalized);
        
        cv::Mat normalized;
        cv::cvtColor(lab_normalized, normalized, cv::COLOR_Lab2BGR);
        
        metrics.psnr_after_norm = calculatePSNR(current, normalized);
        metrics.ssim_after_norm = calculateSSIM(current, normalized);
        
        std::cout << "\n--- Após Normalização ---" << std::endl;
        std::cout << "PSNR: " << metrics.psnr_after_norm << " dB" << std::endl;
        std::cout << "SSIM: " << metrics.ssim_after_norm << std::endl;
        
        current = normalized.clone();
        
        if (save_intermediate) {
            cv::imwrite(output_prefix + "_3_normalized.jpg", current);
        }
        
        // 4. Equalização adaptativa de histograma (CLAHE)
        cv::Mat clahe_result;
        cv::cvtColor(current, lab, cv::COLOR_BGR2Lab);
        cv::split(lab, lab_channels);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);
        
        cv::merge(lab_channels, lab);
        cv::cvtColor(lab, clahe_result, cv::COLOR_Lab2BGR);
        
        metrics.psnr_after_clahe = calculatePSNR(current, clahe_result);
        metrics.ssim_after_clahe = calculateSSIM(current, clahe_result);
        
        std::cout << "\n--- Após CLAHE (Equalização Adaptativa) ---" << std::endl;
        std::cout << "PSNR: " << metrics.psnr_after_clahe << " dB" << std::endl;
        std::cout << "SSIM: " << metrics.ssim_after_clahe << std::endl;
        
        current = clahe_result.clone();
        
        if (save_intermediate) {
            cv::imwrite(output_prefix + "_4_clahe.jpg", current);
        }
        
        // 5. Sharpening suave (realça detalhes das asas)
        cv::Mat sharpened;
        cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0);
        cv::filter2D(current, sharpened, -1, kernel);
        
        // Blend com imagem original (30% sharpening)
        cv::Mat final;
        cv::addWeighted(current, 0.7, sharpened, 0.3, 0, final);
        
        std::cout << "\n--- Após Sharpening ---" << std::endl;
        std::cout << "Aplicado sharpening suave (30% blend)" << std::endl;
        
        if (save_intermediate) {
            cv::imwrite(output_prefix + "_5_sharpened.jpg", final);
        }
        
        std::cout << "\n=== Pré-processamento Concluído ===" << std::endl;
        
        return final;
    }
    
    // Processa lote de imagens
    void batchProcess(const std::string& input_dir, 
                      const std::string& output_dir,
                      const std::string& metrics_file) {
        // Cria diretório de saída se não existir
        fs::create_directories(output_dir);
        
        std::ofstream metrics_csv(metrics_file);
        metrics_csv << "filename,psnr_denoise,ssim_denoise,psnr_norm,ssim_norm,psnr_clahe,ssim_clahe\n";
        
        int count = 0;
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (!entry.is_regular_file()) continue;
            
            std::string ext = entry.path().extension().string();
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".JPG") continue;
            
            std::cout << "\n\n========================================" << std::endl;
            std::cout << "Processando: " << entry.path().filename() << std::endl;
            std::cout << "========================================" << std::endl;
            
            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) {
                std::cerr << "Erro ao ler: " << entry.path() << std::endl;
                continue;
            }
            
            PreprocessingMetrics metrics;
            std::string output_prefix = output_dir + "/" + 
                                       entry.path().stem().string();
            
            cv::Mat processed = preprocess(img, metrics, true, output_prefix);
            
            // Salva imagem final
            cv::imwrite(output_prefix + "_final.jpg", processed);
            
            // Salva métricas
            metrics_csv << entry.path().filename().string() << ","
                       << metrics.psnr_after_denoise << ","
                       << metrics.ssim_after_denoise << ","
                       << metrics.psnr_after_norm << ","
                       << metrics.ssim_after_norm << ","
                       << metrics.psnr_after_clahe << ","
                       << metrics.ssim_after_clahe << "\n";
            
            count++;
        }
        
        metrics_csv.close();
        std::cout << "\n\nTotal de imagens processadas: " << count << std::endl;
        std::cout << "Métricas salvas em: " << metrics_file << std::endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Uso: " << argv[0] << " <input_dir> <output_dir> [metrics_file]" << std::endl;
        std::cout << "Exemplo: " << argv[0] << " ./dataset/train ./preprocessed metrics.csv" << std::endl;
        return -1;
    }
    
    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    std::string metrics_file = (argc > 3) ? argv[3] : "preprocessing_metrics.csv";
    
    ImagePreprocessor preprocessor;
    preprocessor.batchProcess(input_dir, output_dir, metrics_file);
    
    return 0;
}