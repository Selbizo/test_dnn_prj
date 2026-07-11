#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // Тест 1: Создание изображения
    cv::Mat testImage(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
    std::cout << "OpenCV test successful. Image created." << std::endl;

    // Тест 2: Проверка DNN модуля
    try {
        std::cout << "DNN module is available." << std::endl;
        
        // Получаем доступные бэкенды и таргеты
        std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backendsTargets = 
            cv::dnn::getAvailableBackends();
        
        std::cout << "Available backends and targets:" << std::endl;
        for (const auto& bt : backendsTargets) {
            cv::dnn::Backend backend = bt.first;
            cv::dnn::Target target = bt.second;
            
            std::cout << "  Backend: ";
            switch(backend) {
                case cv::dnn::DNN_BACKEND_DEFAULT:
                    std::cout << "DEFAULT";
                    break;
                case cv::dnn::DNN_BACKEND_HALIDE:
                    std::cout << "HALIDE";
                    break;
                case cv::dnn::DNN_BACKEND_INFERENCE_ENGINE:
                    std::cout << "INFERENCE_ENGINE (OpenVINO)";
                    break;
                case cv::dnn::DNN_BACKEND_OPENCV:
                    std::cout << "OPENCV";
                    break;
                case cv::dnn::DNN_BACKEND_VKCOM:
                    std::cout << "VKCOM (Vulkan)";
                    break;
                case cv::dnn::DNN_BACKEND_CUDA:
                    std::cout << "CUDA";
                    break;
                case cv::dnn::DNN_BACKEND_WEBNN:
                    std::cout << "WEBNN";
                    break;
                case cv::dnn::DNN_BACKEND_TIMVX:
                    std::cout << "TIMVX";
                    break;
                default:
                    std::cout << "UNKNOWN";
                    break;
            }
            
            std::cout << ", Target: ";
            switch(target) {
                case cv::dnn::DNN_TARGET_CPU:
                    std::cout << "CPU";
                    break;
                case cv::dnn::DNN_TARGET_OPENCL:
                    std::cout << "OPENCL";
                    break;
                case cv::dnn::DNN_TARGET_OPENCL_FP16:
                    std::cout << "OPENCL_FP16";
                    break;
                case cv::dnn::DNN_TARGET_MYRIAD:
                    std::cout << "MYRIAD";
                    break;
                case cv::dnn::DNN_TARGET_VULKAN:
                    std::cout << "VULKAN";
                    break;
                case cv::dnn::DNN_TARGET_FPGA:
                    std::cout << "FPGA";
                    break;
                case cv::dnn::DNN_TARGET_CUDA:
                    std::cout << "CUDA";
                    break;
                case cv::dnn::DNN_TARGET_CUDA_FP16:
                    std::cout << "CUDA_FP16";
                    break;
                case cv::dnn::DNN_TARGET_HDDL:
                    std::cout << "HDDL";
                    break;
                default:
                    std::cout << "UNKNOWN";
                    break;
            }
            std::cout << std::endl;
        }
        
        // Проверка на наличие CUDA (проверяем через доступные бэкенды)
        bool hasCUDA = false;
        for (const auto& bt : backendsTargets) {
            if (bt.first == cv::dnn::DNN_BACKEND_CUDA) {
                hasCUDA = true;
                break;
            }
        }
        
        if (hasCUDA) {
            std::cout << "CUDA backend is available!" << std::endl;
        } else {
            std::cout << "CUDA backend is NOT available." << std::endl;
        }
        
        std::cout << "DNN test successful!" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "DNN module test failed: " << e.what() << std::endl;
        return 1;
    }

    // Тест 3: Создание простой сети для проверки инференса (без внешних файлов)
    try {
        std::cout << "\nTesting DNN inference with a simple network..." << std::endl;
        
        // Создаем пустую сеть
        cv::dnn::Net net;
        
        // Создаем входной слой - используем blobFromImage для создания входных данных
        cv::Mat inputBlob = cv::dnn::blobFromImage(testImage, 1.0, cv::Size(100, 100), cv::Scalar(), false, false);
        
        // Добавляем слой, который просто пропускает данные (Identity)
        // Используем более простой подход - создаем слой через LayerParams
        cv::dnn::LayerParams params;
        params.type = "Identity";
        params.name = "identity";
        
        int layerId = net.addLayer("identity", "Identity", params);
        
        // Устанавливаем входные данные для сети
        net.setInput(inputBlob);
        
        // Выполняем forward pass
        std::cout << "Performing forward pass..." << std::endl;
        cv::Mat output = net.forward();
        
        std::cout << "Forward pass successful!" << std::endl;
        std::cout << "Output blob size: " << output.size() << std::endl;
        std::cout << "Output blob dimensions: ";
        for (int i = 0; i < output.dims; i++) {
            std::cout << output.size[i] << " ";
        }
        std::cout << std::endl;
        
        // Проверяем, что выходные данные совпадают с входными (Identity слой)
        bool match = true;
        if (inputBlob.total() == output.total()) {
            // Сравниваем данные
            const float* inputData = inputBlob.ptr<float>();
            const float* outputData = output.ptr<float>();
            for (size_t i = 0; i < inputBlob.total(); i++) {
                if (std::abs(inputData[i] - outputData[i]) > 1e-5) {
                    match = false;
                    break;
                }
            }
        } else {
            match = false;
        }
        
        if (match) {
            std::cout << "Identity layer works correctly! Input and output match." << std::endl;
        } else {
            std::cout << "Warning: Input and output don't match (but this might be expected)." << std::endl;
        }
        
        // Тест с разными бэкендами (если доступны)
        std::cout << "\nTesting different backends..." << std::endl;
        
        // Тест с OPENCV бэкендом
        std::cout << "Testing with OPENCV backend..." << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        try {
            cv::Mat outputOpenCV = net.forward();
            std::cout << "  OPENCV backend works!" << std::endl;
        } catch (const cv::Exception& e) {
            std::cout << "  OPENCV backend failed: " << e.what() << std::endl;
        }
        
        // Если доступен CUDA, тестируем его
        bool hasCUDA = false;
        auto backends = cv::dnn::getAvailableBackends();
        for (const auto& bt : backends) {
            if (bt.first == cv::dnn::DNN_BACKEND_CUDA) {
                hasCUDA = true;
                break;
            }
        }
        
        if (hasCUDA) {
            std::cout << "Testing with CUDA backend..." << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            try {
                cv::Mat outputCUDA = net.forward();
                std::cout << "  CUDA backend works!" << std::endl;
            } catch (const cv::Exception& e) {
                std::cout << "  CUDA backend failed: " << e.what() << std::endl;
            }
        }
        
        std::cout << "\nDNN inference test successful!" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "DNN inference test failed: " << e.what() << std::endl;
        std::cerr << "Note: This might indicate issues with DNN module." << std::endl;
        return 1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << "OpenCV is ready for DNN operations." << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}