#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/nbg.h"
#include "tim/vx/types.h"
#include "tim/utils/nbg_parser/nbg_parser.h"
#include "tim/utils/nbg_parser/gc_vip_nbg_format.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <dlfcn.h>
#include <unistd.h>

using namespace cv;
using namespace cv::dnn;

// ========================= MIPI-КАМЕРА =========================
// Камера подключена по MIPI CSI и отдаётся драйвером ARM ISP "juno R2"
// (сенсор OV08A10). Адрес прописан явно, чтобы не зависеть от нумерации
// устройств: /dev/video0 — это и есть MIPI-камера.
static const char* kCameraDevice = "/dev/video0";
static const int kCameraWidth = 1920;
static const int kCameraHeight = 1080;

std::vector<std::string> classes;

void getClasses(std::string classesFile) {
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + classesFile + " not found");
    std::string line;
    while (std::getline(ifs, line))
        classes.push_back(line);
}

void drawPrediction(int classId, float conf, int left, int top, 
                   int right, int bottom, Mat& frame) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    
    std::string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }
    
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
             Point(left + labelSize.width, top + baseLine), 
             Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void yoloPostProcessing(std::vector<Mat>& outs,
                       std::vector<int>& keep_classIds,
                       std::vector<float>& keep_confidences,
                       std::vector<Rect2d>& keep_boxes,
                       float conf_threshold,
                       float iou_threshold,
                       const std::string& model_name,
                       const int nc = 80) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> boxes;
    
    // For YOLOv8/v9/v10, transpose output
    if (model_name == "yolov8" || model_name == "yolov10" || model_name == "yolov9") {
        cv::transposeND(outs[0], {0, 2, 1}, outs[0]);
    }
    
    for (auto preds : outs) {
        preds = preds.reshape(1, preds.size[1]);
        for (int i = 0; i < preds.rows; ++i) {
            // Filter out non-objects
            float obj_conf = (model_name == "yolov8" || model_name == "yolov9" || 
                             model_name == "yolov10") ? 1.0f : preds.at<float>(i, 4);
            if (obj_conf < conf_threshold)
                continue;
            
            Mat scores = preds.row(i).colRange(
                (model_name == "yolov8" || model_name == "yolov9" || 
                 model_name == "yolov10") ? 4 : 5, preds.cols);
            double conf;
            Point maxLoc;
            minMaxLoc(scores, 0, &conf, 0, &maxLoc);
            
            conf = (model_name == "yolov8" || model_name == "yolov9" || 
                   model_name == "yolov10") ? conf : conf * obj_conf;
            if (conf < conf_threshold)
                continue;
            
            // Get bbox coordinates
            float* det = preds.ptr<float>(i);
            double cx = det[0];
            double cy = det[1];
            double w = det[2];
            double h = det[3];
            
            if (model_name == "yolov10") {
                boxes.push_back(Rect2d(cx, cy, w, h));
            } else {
                boxes.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h,
                                      cx + 0.5 * w, cy + 0.5 * h));
            }
            classIds.push_back(maxLoc.x);
            confidences.push_back(static_cast<float>(conf));
        }
    }
    
    // Apply NMS
    std::vector<int> keep_idx;
    NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, keep_idx);
    
    for (auto i : keep_idx) {
        keep_classIds.push_back(classIds[i]);
        keep_confidences.push_back(confidences[i]);
        keep_boxes.push_back(boxes[i]);
    }
}

// INT8-выход YOLOv8n.
//
// Штатный выход output0 — это Concat из координат боксов (0..640) и
// вероятностей классов (0..1). При квантовании на весь тензор берётся один
// масштаб (около 2.5), и вероятности 0..1 превращаются в 0 после округления
// в int8. Поэтому для NPU модель экспортирована с дополнительным выходом
// /model.22/Split_output_1 — это логиты классов до Sigmoid (диапазон
// примерно -20..20), у них свой масштаб. Здесь логиты де-квантуются,
// проходит Sigmoid уже в float, и результат склеивается с боксами в
// обычный тензор 1x84x8400, который понимает yoloPostProcessing().
static Mat mergeYoloOutputs(std::vector<Mat>& outs) {
    Mat* boxOut = nullptr;
    Mat* clsLogits = nullptr;
    for (auto& m : outs) {
        if (m.dims == 3 && m.size[1] == 84)
            boxOut = &m;
        else if (m.dims == 3 && m.size[1] == 80)
            clsLogits = &m;
    }
    if (!boxOut || !clsLogits)
        return outs.empty() ? Mat() : outs[0];

    int n = boxOut->size[2];
    for (int c = 0; c < 80; ++c) {
        float* p = clsLogits->ptr<float>(0, c);
        for (int i = 0; i < n; ++i)
            p[i] = 1.0f / (1.0f + std::exp(-p[i]));
    }

    int sz[] = {1, 84, n};
    Mat merged(3, sz, CV_32F);
    for (int c = 0; c < 4; ++c)
        std::memcpy(merged.ptr<float>(0, c), boxOut->ptr<float>(0, c),
                    n * sizeof(float));
    for (int c = 0; c < 80; ++c)
        std::memcpy(merged.ptr<float>(0, 4 + c), clsLogits->ptr<float>(0, c),
                    n * sizeof(float));
    return merged;
}

// ========================= NPU =========================
// На Banana Pi CM4 (Amlogic A311D) NPU — это Vivante VIP9000, обслуживаемый
// драйвером galcore (/dev/galcore). OpenCV работает с ним через TIM-VX
// (библиотеки в /opt/tim-vx), backend DNN_BACKEND_TIMVX, target DNN_TARGET_NPU.
static const char* backendName(int b) {
    switch (b) {
        case DNN_BACKEND_OPENCV: return "OPENCV";
        case DNN_BACKEND_TIMVX:  return "TIMVX";
        case DNN_BACKEND_CUDA:   return "CUDA";
        case DNN_BACKEND_VKCOM:  return "VKCOM";
        default:                 return "OTHER";
    }
}

static const char* targetName(int t) {
    switch (t) {
        case DNN_TARGET_CPU: return "CPU";
        case DNN_TARGET_NPU: return "NPU";
        case DNN_TARGET_OPENCL: return "OPENCL";
        case DNN_TARGET_OPENCL_FP16: return "OPENCL_FP16";
        default: return "OTHER";
    }
}

// Выставляется в printNpuInfo(): загружен ли патченный libGAL, совместимый
// с ядром galcore 6.4.8. Без него TIM-VX падает, поэтому NPU не включаем.
static bool g_libGalPatched = false;

static bool hasBackend(Backend backend, Target target) {
    for (const auto& item : getAvailableBackends())
        if (item.first == backend && item.second == target)
            return true;
    return false;
}

// Печатает в терминал сведения о NPU: драйвер galcore, доступ к устройству
// и backend'ы, реально собранные в OpenCV.
static bool printNpuInfo() {
    bool timvx = hasBackend(DNN_BACKEND_TIMVX, DNN_TARGET_NPU);

    std::cout << "===== NPU (galcore / TIM-VX) =====\n";
    std::cout << "[NPU] OpenCV: " << CV_VERSION << "\n";
    std::cout << "[NPU] TIM-VX backend в сборке OpenCV: " << (timvx ? "да" : "нет") << "\n";

    std::cout << "[NPU] /dev/galcore: ";
    if (access("/dev/galcore", F_OK) != 0) {
        std::cout << "не найден (модуль galcore не загружен?)\n";
    } else if (access("/dev/galcore", R_OK | W_OK) == 0) {
        std::cout << "доступен для чтения/записи\n";
    } else {
        std::cout << "есть, но нет прав (запуск от root или udev-правило)\n";
    }

    std::ifstream enableNN("/sys/module/galcore/parameters/enableNN");
    std::string nnValue;
    if (enableNN >> nnValue)
        std::cout << "[NPU] galcore enableNN=" << nnValue
                  << " (255 = включены все NPU-ядра)\n";

    // Userspace TIM-VX в /opt/tim-vx имеет версию 6.4.6.345497, а загруженный
    // драйвер galcore — 6.4.8.415784. libGAL жёстко сверяет версии и без
    // патча не создаёт контекст OpenVX. Патченная копия лежит отдельно,
    // подставляется через LD_LIBRARY_PATH (см. run_npu.sh).
    const char* ldp = getenv("LD_LIBRARY_PATH");
    g_libGalPatched = (ldp != nullptr) && (strstr(ldp, "lib-npu-patch") != nullptr);
    std::cout << "[NPU] libGAL.so: "
              << (g_libGalPatched ? "патченная копия (проверка версии обойдена)"
                                  : "системная (/opt/tim-vx/lib)") << "\n";
    if (!g_libGalPatched)
        std::cout << "[NPU] ВНИМАНИЕ: TIM-VX 6.4.6 несовместим с ядром galcore 6.4.8.\n"
                  << "[NPU] NPU не будет включён, чтобы не падать. Запускайте через\n"
                  << "[NPU] ./run_npu.sh — он подставит\n"
                  << "[NPU] /opt/tim-vx/lib-npu-patch/libGAL.so.\n";

    std::cout << "[NPU] Доступные backend/target:\n";
    for (const auto& item : getAvailableBackends())
        std::cout << "[NPU]   " << backendName(item.first)
                  << "/" << targetName(item.second) << "\n";
    return timvx;
}

// ========================= ПОЛНЫЙ NPU-ИНФЕРЕНС (.nb, свой runtime) =========================
// .nb разбирается парсером NBG и исполняется напрямую через TIM-VX.
// Вход и выход остаются в памяти тензора как UINT8: конверсий float->int8/int->float
// нет. Вход модели — 640x640x3 UINT8, planar BGR (CHW), scale=1/255; выходы —
// (144,g,g) UINT8 для g=80/40/20 с affine scale/zero-point из .nb.
class NpuNbEngine {
public:
    enum Kind { YOLO, SSD };

    void setFastResize(bool v) { fastResize_ = v; }
    int inputWidth() const { return inW_; }
    int inputHeight() const { return inH_; }
    Kind kind() const { return kind_; }

    bool init(const std::string& modelPath, const std::string& priorsPath = "") {
        std::ifstream f(modelPath, std::ios::binary);
        if (!f) { err_ = "не открыт " + modelPath; return false; }
        f.seekg(0, std::ios::end);
        std::streamoff sz = f.tellg();
        f.seekg(0, std::ios::beg);
        nbg_.resize((size_t)sz);
        if (!f.read(nbg_.data(), sz)) { err_ = "не прочитан " + modelPath; return false; }

        nbg_parser_data net = NBG_NULL;
        nbg_parser_init(nbg_.data(), (nbg_uint32_t)nbg_.size(), &net);
        if (net == NBG_NULL) { err_ = "парсер NBG не разобрал .nb"; return false; }

        int nin = 0, nout = 0;
        nbg_parser_query_network(net, NBG_PARSER_NETWORK_INPUT_COUNT, &nin, sizeof(nin));
        nbg_parser_query_network(net, NBG_PARSER_NETWORK_OUTPUT_COUNT, &nout, sizeof(nout));
        if (nin != 1 || nout < 1) { err_ = "ожидался 1 вход и >=1 выход"; return false; }
        outNum_ = nout;

        unsigned inDims[MAX_NUM_DIMS] = {0};
        unsigned inDimCount = 0;
        nbg_parser_query_input(net, 0, NBG_PARSER_BUFFER_PROP_NUM_OF_DIMENSION,
                               &inDimCount, sizeof(inDimCount));
        nbg_parser_query_input(net, 0, NBG_PARSER_BUFFER_PROP_DIMENSIONS,
                               inDims, sizeof(inDims) * inDimCount);
        if (inDimCount < 3) { err_ = "неизвестная размерность входа"; return false; }
        float inScale = 0.f;
        int inZp = 0;
        nbg_parser_query_input(net, 0, NBG_PARSER_BUFFER_PROP_SCALE, &inScale, sizeof(inScale));
        nbg_parser_query_input(net, 0, NBG_PARSER_BUFFER_PROP_ZERO_POINT, &inZp, sizeof(inZp));

        ctx_ = tim::vx::Context::Create();
        graph_ = ctx_->CreateGraph();
        op_ = graph_->CreateOperation<tim::vx::ops::NBG>(
            nbg_.data(), (uint32_t)nin, (uint32_t)nout);

        tim::vx::ShapeType inShape(inDims, inDims + inDimCount);
        tim::vx::Quantization inQ(tim::vx::QuantType::ASYMMETRIC, inScale, inZp);
        tim::vx::TensorSpec inSpec(tim::vx::DataType::UINT8, inShape,
                                   tim::vx::TensorAttribute::INPUT, inQ);
        inTensor_ = graph_->CreateTensor(inSpec);
        op_->BindInput(inTensor_);

        for (int i = 0; i < outNum_; ++i) {
            unsigned dims[MAX_NUM_DIMS] = {0};
            unsigned dimCount = 0;
            nbg_parser_query_output(net, i, NBG_PARSER_BUFFER_PROP_NUM_OF_DIMENSION,
                                    &dimCount, sizeof(dimCount));
            nbg_parser_query_output(net, i, NBG_PARSER_BUFFER_PROP_DIMENSIONS,
                                    dims, sizeof(dims) * dimCount);
            float scale = 0.f;
            int zp = 0;
            nbg_parser_query_output(net, i, NBG_PARSER_BUFFER_PROP_SCALE, &scale, sizeof(scale));
            nbg_parser_query_output(net, i, NBG_PARSER_BUFFER_PROP_ZERO_POINT, &zp, sizeof(zp));

            tim::vx::ShapeType shape(dims, dims + dimCount);
            tim::vx::Quantization q(tim::vx::QuantType::ASYMMETRIC, scale, zp);
            tim::vx::TensorSpec spec(tim::vx::DataType::UINT8, shape,
                                     tim::vx::TensorAttribute::OUTPUT, q);
            auto t = graph_->CreateTensor(spec);
            op_->BindOutput(t);
            outTensors_.push_back(t);

            size_t elems = 1;
            for (auto d : shape) elems *= d;
            outBufs_.emplace_back(elems);
            outGrids_.push_back((int)dims[0]);
            outScale_.push_back(scale);
            outZp_.push_back(zp);

            std::cout << "[NPU-NB]   выход[" << i << "]:";
            for (auto d : shape) std::cout << " " << d;
            std::cout << " scale=" << scale << " zp=" << zp << "\n";
        }

        // --- тип модели ---
        // YOLOv8n: 3 выхода по 144*g*g. SSD: 2 выхода 4*N и 91*N.
        if (outNum_ == 2 && outBufs_[0].size() % 4 == 0 &&
            outBufs_[1].size() == (outBufs_[0].size() / 4) * 91) {
            kind_ = SSD;
            ssdN_ = (int)(outBufs_[0].size() / 4);
            inC_ = (int)inDims[0];
            inH_ = (int)inDims[1];
            inW_ = (int)inDims[2];
            std::ifstream pf(priorsPath);
            if (!pf) { err_ = "не открыт " + priorsPath; return false; }
            float v;
            while (pf >> v) priors_.push_back(v);
            if ((int)priors_.size() != 4 * ssdN_) {
                err_ = "box_priors: ожидалось " + std::to_string(4 * ssdN_) +
                       " чисел, получено " + std::to_string(priors_.size());
                return false;
            }
        } else {
            kind_ = YOLO;
            inW_ = (int)inDims[0];
            inH_ = (int)inDims[1];
            inC_ = (int)inDims[2];
        }

        std::cout << "[NPU-NB] модель: " << (kind_ == SSD ? "SSD MobileNet v1" : "YOLOv8n")
                  << ", вход " << inW_ << "x" << inH_ << "x" << inC_
                  << " UINT8, scale=" << inScale << " zp=" << inZp << "\n";

        auto t0 = std::chrono::steady_clock::now();
        if (!graph_->Compile()) { err_ = "TIM-VX graph->Compile() не удался"; return false; }
        double cms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "[NPU-NB] компиляция графа: " << cv::format("%.1f ms", cms) << "\n";

        inBuf_.assign((size_t)inW_ * inH_ * inC_, 0);
        return true;
    }

    const std::string& error() const { return err_; }

    bool infer(const Mat& bgr, std::vector<int>& cls,
               std::vector<float>& conf, std::vector<Rect2d>& boxes) {
        Mat resized;
        resize(bgr, resized, Size(inW_, inH_), 0, 0,
               fastResize_ ? INTER_NEAREST : INTER_LINEAR);
        const unsigned char* px = resized.ptr<unsigned char>();
        const size_t N = (size_t)inW_ * inH_;

        if (kind_ == SSD) {
            // SSD (TensorFlow): HWC RGB, uint8 без конверсий.
            for (size_t i = 0; i < N; ++i) {
                inBuf_[3 * i + 0] = px[3 * i + 2];
                inBuf_[3 * i + 1] = px[3 * i + 1];
                inBuf_[3 * i + 2] = px[3 * i + 0];
            }
        } else {
            // YOLO: planar BGR (CHW).
            for (size_t i = 0; i < N; ++i) {
                inBuf_[i]         = px[3 * i + 0];
                inBuf_[N + i]     = px[3 * i + 1];
                inBuf_[2 * N + i] = px[3 * i + 2];
            }
        }

        if (!inTensor_->CopyDataToTensor(inBuf_.data()))
            return false;
        if (!graph_->Run())
            return false;
        for (int i = 0; i < outNum_; ++i)
            if (!outTensors_[i]->CopyDataFromTensor(outBufs_[i].data()))
                return false;

        return kind_ == SSD ? postSsd(cls, conf, boxes)
                            : postYolo(cls, conf, boxes);
    }

private:
    bool postYolo(std::vector<int>& cls, std::vector<float>& conf,
                  std::vector<Rect2d>& boxes) {
        cls.clear(); conf.clear(); boxes.clear();
        struct Cand { float score; int cls; float x1, y1, x2, y2; };
        std::vector<Cand> cands;
        const float kLogitThr = std::log(0.4f / 0.6f);

        for (int oi = 0; oi < outNum_; ++oi) {
            const int g = outGrids_[oi];
            const size_t plane = (size_t)g * g;
            const float qScale = outScale_[oi];
            const float qZp = (float)outZp_[oi];
            const unsigned char* d = outBufs_[oi].data();
            const float stride = (float)inW_ / g;
            const float rawThr = qZp + kLogitThr / qScale;

            for (int y = 0; y < g; ++y) {
                for (int x = 0; x < g; ++x) {
                    const size_t base = (size_t)y * g + x;
                    unsigned char bestRaw = 0;
                    int bestCls = -1;
                    for (int c = 0; c < 80; ++c) {
                        unsigned char v = d[(size_t)c * plane + base];
                        if (v > bestRaw) { bestRaw = v; bestCls = c; }
                    }
                    if ((float)bestRaw < rawThr)
                        continue;
                    const float bestLogit = ((float)bestRaw - qZp) * qScale;
                    const float best = 1.f / (1.f + std::exp(-bestLogit));

                    float dist[4];
                    for (int j = 0; j < 4; ++j) {
                        float m = -1e30f;
                        for (int k = 0; k < 16; ++k) {
                            float v = ((float)d[(size_t)(80 + 16 * j + k) * plane + base] - qZp) * qScale;
                            m = std::max(m, v);
                        }
                        float sum = 0.f, acc = 0.f;
                        for (int k = 0; k < 16; ++k) {
                            float v = ((float)d[(size_t)(80 + 16 * j + k) * plane + base] - qZp) * qScale;
                            float e = std::exp(v - m);
                            sum += e;
                            acc += e * k;
                        }
                        dist[j] = acc / std::max(sum, 1e-12f);
                    }

                    float cx = (x + 0.5f) * stride;
                    float cy = (y + 0.5f) * stride;
                    cands.push_back({best, bestCls,
                                     cx - dist[0] * stride, cy - dist[1] * stride,
                                     cx + dist[2] * stride, cy + dist[3] * stride});
                }
            }
        }
        return nmsAndEmit(cands, cls, conf, boxes, 0.45f, /*byClass=*/true);
    }

    bool postSsd(std::vector<int>& cls, std::vector<float>& conf,
                 std::vector<Rect2d>& boxes) {
        cls.clear(); conf.clear(); boxes.clear();
        const int N = ssdN_;
        const int NC = 91;
        const unsigned char* loc = outBufs_[0].data();
        const unsigned char* cl = outBufs_[1].data();
        const float locSc = outScale_[0], locZp = (float)outZp_[0];
        const float clsSc = outScale_[1], clsZp = (float)outZp_[1];
        const float kLogitThr = std::log(0.3f / 0.7f);
        const float rawThr = clsZp + kLogitThr / clsSc;

        struct Cand { float score; int cls; float x1, y1, x2, y2; };
        std::vector<Cand> cands;

        for (int i = 0; i < N; ++i) {
            unsigned char bestRaw = 0;
            int bestC = -1;
            for (int c = 1; c < NC; ++c) {
                unsigned char v = cl[(size_t)i * NC + c];
                if (v > bestRaw) { bestRaw = v; bestC = c; }
            }
            if ((float)bestRaw < rawThr)
                continue;
            const float score = 1.f / (1.f + std::exp(-((float)bestRaw - clsZp) * clsSc));

            auto L = [&](int k) {
                return ((float)loc[(size_t)i * 4 + k] - locZp) * locSc;
            };
            const float yc = L(0), xc = L(1), hh = L(2), ww = L(3);
            const float py = priors_[0 * N + i], px = priors_[1 * N + i];
            const float ph = priors_[2 * N + i], pw = priors_[3 * N + i];

            float ycenter = yc / 10.f * ph + py;
            float xcenter = xc / 10.f * pw + px;
            float h = std::exp(hh / 5.f) * ph;
            float w = std::exp(ww / 5.f) * pw;
            float x1 = std::max(0.f, std::min(1.f, xcenter - w / 2));
            float y1 = std::max(0.f, std::min(1.f, ycenter - h / 2));
            float x2 = std::max(0.f, std::min(1.f, xcenter + w / 2));
            float y2 = std::max(0.f, std::min(1.f, ycenter + h / 2));
            cands.push_back({score, bestC, x1 * inW_, y1 * inH_, x2 * inW_, y2 * inH_});
        }
        // SSD-пример Khadas подавляет боксы без учёта класса.
        return nmsAndEmit(cands, cls, conf, boxes, 0.45f, /*byClass=*/false);
    }

    template <typename Cand>
    bool nmsAndEmit(std::vector<Cand>& cands, std::vector<int>& cls,
                    std::vector<float>& conf, std::vector<Rect2d>& boxes,
                    float iouThr, bool byClass) {
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.score > b.score; });
        std::vector<Cand> keep;
        for (const auto& c : cands) {
            bool suppressed = false;
            for (const auto& k : keep) {
                if (byClass && k.cls != c.cls)
                    continue;
                float xx1 = std::max(c.x1, k.x1), yy1 = std::max(c.y1, k.y1);
                float xx2 = std::min(c.x2, k.x2), yy2 = std::min(c.y2, k.y2);
                float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
                float inter = w * h;
                float a1 = (c.x2 - c.x1) * (c.y2 - c.y1);
                float a2 = (k.x2 - k.x1) * (k.y2 - k.y1);
                if (inter / std::max(a1 + a2 - inter, 1e-9f) > iouThr) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed)
                keep.push_back(c);
        }
        for (const auto& k : keep) {
            cls.push_back(k.cls);
            conf.push_back(k.score);
            boxes.emplace_back(k.x1, k.y1, k.x2, k.y2);
        }
        return true;
    }

    std::vector<char> nbg_;
    std::shared_ptr<tim::vx::Context> ctx_;
    std::shared_ptr<tim::vx::Graph> graph_;
    std::shared_ptr<tim::vx::ops::NBG> op_;
    std::shared_ptr<tim::vx::Tensor> inTensor_;
    std::vector<std::shared_ptr<tim::vx::Tensor>> outTensors_;
    std::vector<unsigned char> inBuf_;
    std::vector<std::vector<unsigned char>> outBufs_;
    std::vector<int> outGrids_;
    std::vector<float> outScale_;
    std::vector<int> outZp_;
    std::vector<float> priors_;
    Kind kind_ = YOLO;
    int ssdN_ = 0;
    int inW_ = 640, inH_ = 640, inC_ = 3;
    int outNum_ = 0;
    bool fastResize_ = false;
    std::string err_;
};

// ========================= ПОТОК ЗАХВАТА =========================
// Отдельный поток непрерывно читает камеру и держит только ПОСЛЕДНИЙ кадр.
// Конверсия YUYV->BGR, инференс и отображение выполняются в потоке обработки,
// поэтому захват не тормозится тяжёлыми операциями.
//
// Почему "сырое" чтение: с CAP_PROP_CONVERT_RGB=0 OpenCV не конвертирует
// YUYV->BGR внутри capture-потока (это ~7 мс на кадр 1080p). Без конверсии
// поток захвата успевает ~117–120 fps, с конверсией — ~76–99 fps.
struct CameraCapture {
    VideoCapture cap;
    int width = kCameraWidth;
    int height = kCameraHeight;
    bool grey = false;

    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    Mat latest;          // последний полученный кадр (сырые байты)
    uint64_t seq = 0;    // номер последнего кадра
    std::atomic<bool> stop{false};
    std::atomic<long> captured{0};
    int targetFps = 0;           // 0 = без ограничения
    std::atomic<int> readFailures{0};

    bool open(int w, int h, bool useGrey) {
        grey = useGrey;
        cap.open(kCameraDevice, CAP_V4L2);
        if (!cap.isOpened())
            return false;
        cap.set(CAP_PROP_FRAME_WIDTH, w);
        cap.set(CAP_PROP_FRAME_HEIGHT, h);
        cap.set(CAP_PROP_FOURCC,
                VideoWriter::fourcc(grey ? 'G' : 'Y', grey ? 'R' : 'U',
                                    grey ? 'E' : 'Y', grey ? 'Y' : 'V'));
        // Сырые байты: конверсию делаем сами и только для нужных кадров.
        cap.set(CAP_PROP_CONVERT_RGB, 0);
        // Больше буферов = меньше пропусков при 120 fps.
        cap.set(CAP_PROP_BUFFERSIZE, 8);
        width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        return width > 0 && height > 0;
    }

    // Однократное чтение (нужно для калибровки INT8, пока поток не запущен).
    bool readOnce(Mat& out) {
        if (!cap.grab())
            return false;
        if (!cap.retrieve(out))
            return false;
        return !out.empty();
    }

    // Сырые байты -> BGR сразу нужного размера: ресайз YUYV/GREY и конверсия
    // только маленького кадра. Экономит ~6 мс на 1080p.
    bool toBgrScaled(const Mat& raw, Mat& bgr, int outW, int outH, bool nearest) const {
        const size_t expected = static_cast<size_t>(width) * height * (grey ? 1 : 2);
        if (raw.total() * raw.elemSize() < expected)
            return false;
        const int interp = nearest ? INTER_NEAREST : INTER_LINEAR;
        if (grey) {
            Mat g(height, width, CV_8UC1, const_cast<uchar*>(raw.data));
            Mat small;
            resize(g, small, Size(outW, outH), 0, 0, interp);
            cvtColor(small, bgr, COLOR_GRAY2BGR);
        } else {
            Mat yuyv(height, width, CV_8UC2, const_cast<uchar*>(raw.data));
            Mat small;
            resize(yuyv, small, Size(outW, outH), 0, 0, interp);
            cvtColor(small, bgr, COLOR_YUV2BGR_YUYV);
        }
        return true;
    }

    // Сырые байты -> BGR (в потоке обработки).
    bool toBgr(const Mat& raw, Mat& bgr) const {
        const size_t expected = static_cast<size_t>(width) * height * (grey ? 1 : 2);
        if (raw.total() * raw.elemSize() < expected)
            return false;
        if (grey) {
            Mat g(height, width, CV_8UC1, const_cast<uchar*>(raw.data));
            cvtColor(g, bgr, COLOR_GRAY2BGR);
        } else {
            Mat yuyv(height, width, CV_8UC2, const_cast<uchar*>(raw.data));
            cvtColor(yuyv, bgr, COLOR_YUV2BGR_YUYV);
        }
        return true;
    }

    void start() {
        stop = false;
        th = std::thread(&CameraCapture::run, this);
    }

    void run() {
        Mat raw;
        auto nextDue = std::chrono::steady_clock::now();
        while (!stop.load()) {
            if (!cap.grab()) {
                if (++readFailures > 100)
                    break;
                continue;
            }
            if (!cap.retrieve(raw) || raw.empty())
                continue;
            readFailures = 0;
            {
                std::lock_guard<std::mutex> lk(m);
                // Обмен, а не копирование: буферы переиспользуются.
                std::swap(latest, raw);
                ++seq;
            }
            cv.notify_one();
            ++captured;

            // Ограничение частоты: освобождает память/CPU для NPU.
            if (targetFps > 0) {
                nextDue += std::chrono::microseconds(1000000 / targetFps);
                std::this_thread::sleep_until(nextDue);
                auto now = std::chrono::steady_clock::now();
                if (nextDue < now - std::chrono::milliseconds(100))
                    nextDue = now;
            }
        }
    }

    // Ждём кадр новее lastSeq. false — таймаут или остановка.
    bool waitNext(Mat& out, uint64_t& lastSeq, int timeoutMs = 1000) {
        std::unique_lock<std::mutex> lk(m);
        if (!cv.wait_for(lk, std::chrono::milliseconds(timeoutMs),
                         [&] { return seq != lastSeq || stop.load(); }))
            return false;
        if (seq == lastSeq)
            return false;
        std::swap(out, latest);
        lastSeq = seq;
        return true;
    }

    void shutdown() {
        stop = true;
        cv.notify_all();
        if (th.joinable())
            th.join();
        cap.release();
    }
};

static bool startsWith(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

int main(int argc, char* argv[]) {
    bool use_npu = false;
    bool useNb = false;          // полный NPU: Acuity .nb, все слои на NPU
    bool useSsd = false;         // вместо YOLO — MobileNet-SSD 300
    double runSeconds = 0.0;
    bool estimateOnly = false;   // пиковый fps: захват + потребление без обработки
    bool noGui = false;
    bool noDetect = false;
    int detectEvery = 50;        // инференс на каждый N-й обработанный кадр
    bool useGrey = false;        // GREY 1080p — самый быстрый формат (~120 fps)
    int viewWidth = 0;           // ширина окна (0 = полный кадр)
    bool fastResize = false;     // INTER_NEAREST вместо LINEAR
    int captureFps = 0;          // 0 = максимальная частота захвата

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--npu" || arg == "-n") {
            use_npu = true;
        } else if (arg == "--npu-nb") {
            use_npu = true;
            useNb = true;
        } else if (arg == "--ssd") {
            use_npu = true;
            useNb = true;
            useSsd = true;
        } else if (arg == "--cpu" || arg == "-c") {
            use_npu = false;
            useNb = false;
        } else if (arg == "--duration" && i + 1 < argc) {
            runSeconds = atof(argv[++i]);
        } else if (arg == "--estimate-only") {
            estimateOnly = true;
            noGui = true;
        } else if (arg == "--no-gui") {
            noGui = true;
        } else if (arg == "--no-detect") {
            noDetect = true;
        } else if (arg == "--detect-every" && i + 1 < argc) {
            detectEvery = std::max(1, atoi(argv[++i]));
        } else if (arg == "--grey") {
            useGrey = true;
        } else if (arg == "--capture-fps" && i + 1 < argc) {
            captureFps = std::max(0, atoi(argv[++i]));
        } else if (arg == "--fast-resize") {
            fastResize = true;
        } else if (arg == "--view-width" && i + 1 < argc) {
            viewWidth = std::max(0, atoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: facedetection [options]\n"
                << "  --npu, -n          Inference on NPU (TIM-VX, INT8; лучше через ./run_npu.sh)\n"
                << "  --npu-nb           Полный NPU: Acuity .nb, все слои и операции на NPU\n"
                << "                     (tools/npu/yolov8n_uint8.nb; запускать через ./run_npu.sh)\n"
                << "  --ssd              Полный NPU, MobileNet-SSD 300 (быстрее, но менее точно)\n"
                << "  --cpu, -c          CPU backend (default)\n"
                << "  --duration <sec>   Stop after the given number of seconds\n"
                << "  --estimate-only    Пиковый fps: два потока (захват + потребитель),\n"
                << "                     без конверсии, инференса и окна\n"
                << "  --no-gui           Без окна\n"
                << "  --no-detect        Без инференса (только захват + показ)\n"
                << "  --detect-every <N> Инференс на каждый N-й кадр (default 50)\n"
                << "  --grey             Захват GREY 1080p вместо YUYV (до ~120 fps)\n"
                << "  --capture-fps <N>  Ограничить захват N кадров/с (0 = максимум).\n"
                << "                     30 fps освобождает память для NPU\n"
                << "  --fast-resize      resize 640x640 через INTER_NEAREST (быстрее)\n"
                << "  --view-width <px>  Ширина окна (0 = полный кадр); 960 заметно\n"
                << "                     разгружает показ и сохраняет высокий fps\n"
                << "  --help, -h         Show this help\n"
                << "Camera: " << kCameraDevice << " (MIPI, "
                << kCameraWidth << "x" << kCameraHeight << ")\n";
            return 0;
        } else {
            std::cerr << "Неизвестный аргумент: " << arg << "\n";
            return 2;
        }
    }

    // Load model
    std::filesystem::path exeDir = std::filesystem::canonical("/proc/self/exe").parent_path();
    std::string weightPath = (exeDir / "../../src/yolov8n.onnx").string();
    bool npuModelLoaded = false;
    if (use_npu && !useNb) {
        std::string npuPath = (exeDir / "../../src/yolov8n_npu.onnx").string();
        if (std::filesystem::exists(npuPath)) {
            weightPath = npuPath;
            npuModelLoaded = true;
            std::cout << "[NPU] Модель с выходом логитов классов: " << npuPath << "\n";
        } else {
            std::cout << "[NPU] Не найден yolov8n_npu.onnx — модель с отдельным\n"
                      << "[NPU] выходом логитов классов. Без неё INT8 теряет классы.\n"
                      << "[NPU] NPU не будет использован.\n";
        }
    }
    Net net;
    if (!estimateOnly && !useNb)
        net = readNet(weightPath);

    bool npuActive = false;
    bool npuRequested = false;
    if (useNb && !estimateOnly) {
        // Полный NPU-путь: OpenCV не нужен, но патченный libGAL и root
        // нужны так же, как для TIM-VX.
        printNpuInfo();
        if (!g_libGalPatched) {
            std::cerr << "[NPU-NB] Нужен патченный libGAL. Запускайте через\n"
                      << "[NPU-NB] ./run_npu.sh --npu-nb ...\n";
            return 1;
        }
        if (access("/dev/galcore", R_OK | W_OK) != 0) {
            std::cerr << "[NPU-NB] Нет доступа к /dev/galcore (нужен root).\n";
            return 1;
        }
    }
    if (use_npu && !estimateOnly && !useNb) {
        bool timvxAvailable = printNpuInfo();
#if defined(HAVE_OPENCV_DNN_AIPU)
        // Патченный Amlogic OpenCV с собственным AIPU-бэкендом (на всякий случай).
        net.setPreferableBackend(DNN_BACKEND_AIPU);
        net.setPreferableTarget(DNN_TARGET_AIPU);
        npuActive = true;
        std::cout << "[NPU] Используется backend AIPU\n";
#else
        // TIM-VX в OpenCV 4.12 умеет только INT8-слои, поэтому модель нужно
        // квантовать. Калибровка — по src/image.jpg (или кадру камеры, если
        // картинки нет), уже после открытия камеры.
        npuRequested = timvxAvailable && g_libGalPatched && npuModelLoaded &&
                       access("/dev/galcore", R_OK | W_OK) == 0;
        if (timvxAvailable && !g_libGalPatched) {
            std::cout << "[NPU] TIM-VX не включён: нужен патченный libGAL.\n"
                      << "[NPU] Используйте ./run_npu.sh --npu ...\n"
                      << "[NPU] Пока используется CPU.\n";
        } else if (timvxAvailable && !npuModelLoaded) {
            std::cout << "[NPU] TIM-VX не включён: нет yolov8n_npu.onnx.\n"
                      << "[NPU] Пока используется CPU.\n";
        } else if (timvxAvailable && !npuRequested) {
            std::cout << "[NPU] TIM-VX есть, но нет доступа к /dev/galcore.\n"
                      << "[NPU] Запустите от root (sudo) или добавьте udev-правило.\n"
                      << "[NPU] Пока используется CPU.\n";
        } else if (!timvxAvailable) {
            std::cout << "[NPU] TIM-VX недоступен в этой сборке OpenCV — используется CPU\n";
        }
#endif
    }

    // Load classes
    std::string classPath = (exeDir / (useSsd
        ? "../../tools/npu/ssd_labels.txt"
        : "../../src/object_detection_classes_yolo.txt")).string();
    getClasses(classPath);
    if (useSsd)
        std::cout << "[NPU-NB] SSD-классов: " << classes.size() << "\n";

    // Open MIPI camera (V4L2). Адрес задан строкой, а не индексом: индекс 0
    // на этой плате может указывать не на MIPI-камеру, а на другой узел.
    CameraCapture camera;
    if (!camera.open(kCameraWidth, kCameraHeight, useGrey)) {
        std::cerr << "[ERROR] Не удалось открыть MIPI-камеру " << kCameraDevice << "\n";
        return -1;
    }
    camera.targetFps = captureFps;
    std::cout << "[INFO] MIPI-камера " << kCameraDevice << ": "
              << camera.width << "x" << camera.height
              << " (" << (useGrey ? "GREY" : "YUYV")
              << ", сырое чтение, 2 потока)\n";

    // ===== Полный NPU: Acuity .nb =====
    NpuNbEngine nb;
    bool nbActive = false;
    if (useNb && !estimateOnly) {
        std::string nbModel = (exeDir / (useSsd
            ? "../../tools/npu/mobilenet_ssd_uint8.nb"
            : "../../tools/npu/yolov8n_uint8.nb")).string();
        std::string nbPriors = useSsd
            ? (exeDir / "../../tools/npu/box_priors.txt").string() : "";
        if (!std::filesystem::exists(nbModel)) {
            std::cerr << "[NPU-NB] Не найден файл: " << nbModel << "\n";
            return 1;
        }
        std::cout << "[NPU-NB] Модель: " << nbModel << "\n";
        if (!nb.init(nbModel, nbPriors)) {
            std::cerr << "[NPU-NB] Ошибка инициализации: " << nb.error() << "\n";
            return 1;
        }
        nb.setFastResize(fastResize);
        nbActive = true;
        std::cout << "[NPU-NB] Весь граф .nb исполняется на NPU, "
                     "слоёв на CPU нет\n";
    }

    // ===== Подготовка INT8 для TIM-VX/NPU =====
    bool int8Mode = false;
    std::vector<float> outputScale;
    std::vector<int> outputZp;
    float inputScale = 1.f;
    int inputZp = 0;

    if (npuRequested) {
        try {
            // Калибровка по src/image.jpg. Важно: если брать первый кадр
            // камеры, тёмная сцена даёт input scale меньше 1/255, и на
            // светлой сцене вход будет клиппироваться. image.jpg имеет
            // полный диапазон 0..255, поэтому input scale получается 1/255.
            // Если картинки нет — берём кадр камеры.
            std::string calibPath = (exeDir / "../../src/image.jpg").string();
            Mat calib = imread(calibPath);
            if (!calib.empty()) {
                std::cout << "[NPU] Калибровка по " << calibPath << "\n";
            } else {
                Mat raw;
                if (camera.readOnce(raw))
                    camera.toBgr(raw, calib);
                if (calib.empty())
                    throw std::runtime_error("нет image.jpg и камера не отдала кадр");
                std::cout << "[NPU] Калибровка по кадру камеры\n";
            }
            if (calib.channels() != 3)
                cvtColor(calib, calib, COLOR_GRAY2BGR);
            Mat calibBlob = blobFromImage(calib, 1.0 / 255.0, Size(640, 640),
                                          Scalar(), true, false);

            std::cout << "[NPU] Квантование модели в INT8...\n";
            auto qStart = std::chrono::steady_clock::now();
            net = net.quantize({calibBlob}, CV_8S, CV_8S, true);
            double qMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - qStart).count();
            std::cout << "[NPU] quantize: " << cv::format("%.1f ms", qMs) << "\n";

            std::vector<float> inScale;
            std::vector<int> inZp;
            net.getInputDetails(inScale, inZp);
            net.getOutputDetails(outputScale, outputZp);
            inputScale = inScale[0];
            inputZp = inZp[0];

            auto qOutNames = net.getUnconnectedOutLayersNames();
            if (qOutNames.size() < 2)
                throw std::runtime_error(
                    "ожидались 2 выхода (output0 + логиты классов), получено " +
                    std::to_string(qOutNames.size()));

            int tvLayers = 0, totalLayers = 0;
            for (const auto& name : net.getLayerNames()) {
                auto layer = net.getLayer(net.getLayerId(name));
                totalLayers++;
                if (layer->supportBackend(DNN_BACKEND_TIMVX))
                    tvLayers++;
            }
            std::cout << "[NPU] INT8: input scale=" << inputScale
                      << " zp=" << inputZp
                      << ", output scale=" << outputScale[0]
                      << " zp=" << outputZp[0] << "\n";
            std::cout << "[NPU] Слоёв с поддержкой TIM-VX: "
                      << tvLayers << "/" << totalLayers
                      << " (остальные выполняются на CPU)\n";

            net.setPreferableBackend(DNN_BACKEND_TIMVX);
            net.setPreferableTarget(DNN_TARGET_NPU);
            int8Mode = true;
            npuActive = true;
            std::cout << "[NPU] Используется backend TIM-VX, target NPU (INT8)\n";
        } catch (const std::exception& e) {
            std::cerr << "[NPU] Не удалось подготовить INT8 для NPU: "
                      << e.what() << "\n";
            std::cout << "[NPU] Откат на CPU\n";
        }
    }

    if (!npuActive && !estimateOnly && !useNb) {
        // Если загружали NPU-модель с двумя выходами, но NPU не включился,
        // возвращаемся к обычной yolov8n.onnx: у неё один выход, и CPU-путь
        // не пытается обрабатывать логиты как готовые детекции.
        if (npuModelLoaded) {
            net = readNet((exeDir / "../../src/yolov8n.onnx").string());
            std::cout << "[NPU] Откат на обычную модель yolov8n.onnx\n";
        }
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        std::cout << "[INFO] Using CPU backend\n";
    }

    std::vector<String> outputNames;
    if (!estimateOnly && !useNb)
        outputNames = net.getUnconnectedOutLayersNames();

    if (estimateOnly)
        std::cout << "[EST] Режим пикового fps: захват + потребитель, "
                     "без конверсии/инференса/окна\n";

    // ---------------- Запуск потока захвата ----------------
    auto runStart = std::chrono::steady_clock::now();
    camera.start();

    Mat raw, img;
    uint64_t lastSeq = 0;
    long processed = 0;
    int detectCount = 0;
    double detectTotalMs = 0.0;
    std::vector<Rect2d> prev_boxes;
    std::vector<int> prev_classIds;
    std::vector<float> prev_confidences;

    long capFrames = 0, pipeFrames = 0;
    double capFps = 0.0, pipeFps = 0.0;
    auto statStart = std::chrono::steady_clock::now();

    bool running = true;
    while (running) {
        if (runSeconds > 0.0) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - runStart).count();
            if (elapsed >= runSeconds) {
                std::cout << "[INFO] Задано " << runSeconds << " с — останавливаемся\n";
                break;
            }
        }

        // Потребитель берёт только самый свежий кадр: старые пропускаются.
        if (!camera.waitNext(raw, lastSeq, 1000))
            continue;

        ++pipeFrames;

        // В режиме пикового fps ничего не делаем с кадром — измеряем предел
        // конвейера «захват -> потребитель».
        if (!estimateOnly) {
            if (nbActive && noGui) {
                // Без окна полный 1080p BGR не нужен: сразу уменьшаем YUYV.
                if (!camera.toBgrScaled(raw, img, nb.inputWidth(), nb.inputHeight(), fastResize))
                    continue;
            } else if (!camera.toBgr(raw, img)) {
                continue;
            }

            std::vector<int> keep_classIds;
            std::vector<float> keep_confidences;
            std::vector<Rect2d> keep_boxes;

            if (nbActive) {
                // Полный NPU: каждый кадр прогоняется через .nb.
                if (!noDetect) {
                    auto fwdStart = std::chrono::steady_clock::now();
                    if (!nb.infer(img, keep_classIds, keep_confidences, keep_boxes)) {
                        std::cerr << "[NPU-NB] Ошибка инференса\n";
                    } else {
                        // Боксы приходят в координатах сети 640x640.
                        double sx = (double)img.cols / nb.inputWidth();
                        double sy = (double)img.rows / nb.inputHeight();
                        for (auto& b : keep_boxes)
                            b = Rect2d(b.x * sx, b.y * sy,
                                       b.width * sx, b.height * sy);
                    }
                    double fwdMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - fwdStart).count();
                    detectCount++;
                    detectTotalMs += fwdMs;
                    std::cout << "[NPU-NB] forward #" << detectCount << ": "
                              << cv::format("%.1f ms", fwdMs)
                              << " (avg " << cv::format("%.1f ms", detectTotalMs / detectCount)
                              << "), объектов: " << keep_boxes.size();
                    if (!keep_classIds.empty() && !classes.empty())
                        std::cout << " (первый: " << classes[keep_classIds[0]]
                                  << " " << cv::format("%.2f", keep_confidences[0]) << ")";
                    std::cout << "\n";
                    prev_boxes = keep_boxes;
                    prev_classIds = keep_classIds;
                    prev_confidences = keep_confidences;
                } else {
                    keep_boxes = prev_boxes;
                    keep_classIds = prev_classIds;
                    keep_confidences = prev_confidences;
                }
            } else {
            bool run_detection = !noDetect && (processed % detectEvery == 0);

            if (run_detection) {
                if (npuActive && int8Mode && detectCount == 0) {
                    std::cout << "[NPU] Первый forward собирает граф TIM-VX — "
                                 "это может занять несколько десятков секунд...\n";
                }
                // Preprocess
                Mat blob;
                blobFromImage(img, blob, 1.0/255.0, Size(640, 640),
                              Scalar(), true, false);

                // Для INT8-модели вход тоже INT8: те же 0..1, но с масштабом/zero-point,
                // которые OpenCV выбрал при квантовании.
                Mat netInput = blob;
                if (npuActive && int8Mode)
                    blob.convertTo(netInput, CV_8S, 1.f / inputScale, inputZp);

                auto fwdStart = std::chrono::steady_clock::now();
                net.setInput(netInput);
                std::vector<Mat> outs;
                net.forward(outs, outputNames);

                // TIM-VX возвращает INT8-выходы. Переводим их обратно в float,
                // иначе постобработка (minMaxLoc и т.п.) не сможет их прочитать.
                // Затем логиты классов пропускаем через Sigmoid и склеиваем с
                // боксами в один тензор 1x84x8400.
                if (npuActive && int8Mode) {
                    for (size_t oi = 0; oi < outs.size(); ++oi) {
                        if (outs[oi].type() != CV_8S)
                            continue;
                        float s = (oi < outputScale.size()) ? outputScale[oi]
                                                             : outputScale[0];
                        int zp = (oi < outputZp.size()) ? outputZp[oi]
                                                        : outputZp[0];
                        Mat f;
                        outs[oi].convertTo(f, CV_32F, s, -s * zp);
                        outs[oi] = f;
                    }
                    Mat merged = mergeYoloOutputs(outs);
                    outs.clear();
                    outs.push_back(merged);
                }
                double fwdMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - fwdStart).count();
                detectCount++;
                detectTotalMs += fwdMs;
                std::cout << (npuActive ? "[NPU]" : "[CPU]")
                          << " forward #" << detectCount << ": "
                          << cv::format("%.1f ms", fwdMs)
                          << " (avg " << cv::format("%.1f ms", detectTotalMs / detectCount)
                          << ")\n";

                yoloPostProcessing(outs, keep_classIds, keep_confidences, keep_boxes,
                                  0.3, 0.2, "yolov8", 80);
                std::cout << (npuActive ? "[NPU]" : "[CPU]")
                          << " объектов после NMS: " << keep_boxes.size();
                if (!keep_classIds.empty() && !classes.empty())
                    std::cout << " (первый: " << classes[keep_classIds[0]]
                              << " " << cv::format("%.2f", keep_confidences[0]) << ")";
                std::cout << "\n";

                prev_boxes = keep_boxes;
                prev_classIds = keep_classIds;
                prev_confidences = keep_confidences;
            } else {
                keep_boxes = prev_boxes;
                keep_classIds = prev_classIds;
                keep_confidences = prev_confidences;
            }
            } // end else: OpenCV CPU / TIM-VX

            if (!noGui) {
                // Draw results
                for (size_t idx = 0; idx < keep_boxes.size(); ++idx) {
                    Rect box(cvFloor(keep_boxes[idx].x), cvFloor(keep_boxes[idx].y),
                             cvFloor(keep_boxes[idx].width - keep_boxes[idx].x),
                             cvFloor(keep_boxes[idx].height - keep_boxes[idx].y));
                    drawPrediction(keep_classIds[idx], keep_confidences[idx],
                                   box.x, box.y, box.width + box.x,
                                   box.height + box.y, img);
                }

                putText(img, format("MIPI %s  cap %.1f  pipe %.1f fps",
                                    kCameraDevice, capFps, pipeFps),
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 255, 0), 2);
                // Показ полного 1080p стоит ~40 мс на кадр и просаживает
                // захват до ~85 fps. Уменьшенное окно оставляет ~115-120 fps.
                Mat view = img;
                if (viewWidth > 0 && img.cols > viewWidth) {
                    Mat small;
                    int vh = static_cast<int>(
                        std::lround(static_cast<double>(img.rows) * viewWidth / img.cols));
                    resize(img, small, Size(viewWidth, vh));
                    view = small;
                }
                imshow("YOLO Object Detection", view);
                int key = waitKey(1);
                if (key == 'q' || key == 27)
                    running = false;
            }
        }

        ++processed;

        // Раз в секунду печатаем фактическую частоту захвата и конвейера.
        auto now = std::chrono::steady_clock::now();
        double statElapsed = std::chrono::duration<double>(now - statStart).count();
        if (statElapsed >= 1.0) {
            long capTotal = camera.captured.load();
            capFps = (capTotal - capFrames) / statElapsed;
            pipeFps = pipeFrames / statElapsed;
            capFrames = capTotal;
            pipeFrames = 0;
            statStart = now;

            std::cout << (estimateOnly ? "[EST] " : "[PERF] ")
                      << "capture " << cv::format("%.1f", capFps) << " fps"
                      << "  |  pipeline " << cv::format("%.1f", pipeFps) << " fps"
                      << "  |  dropped " << (capTotal - processed)
                      << "  |  detects " << detectCount << "\n";
        }
    }

    auto runEnd = std::chrono::steady_clock::now();
    camera.shutdown();

    double runSecs = std::chrono::duration<double>(runEnd - runStart).count();
    long totalCap = camera.captured.load();
    std::cout << "[INFO] Кадров захвачено: " << totalCap
              << ", обработано: " << processed << "\n";
    std::cout << "[INFO] Средняя частота захвата: "
              << cv::format("%.1f", runSecs > 0 ? totalCap / runSecs : 0.0)
              << " fps"
              << (estimateOnly ? " (end-to-end)" : "") << "\n";
    if (!estimateOnly)
        std::cout << "[INFO] Средняя частота конвейера: "
                  << cv::format("%.1f", runSecs > 0 ? processed / runSecs : 0.0)
                  << " fps\n";
    if (detectCount > 0) {
        std::cout << "[INFO] Инференсов: " << detectCount
                  << ", среднее время forward: "
                  << cv::format("%.1f ms", detectTotalMs / detectCount)
                  << " (" << (nbActive ? "полный NPU (.nb)"
                                       : (npuActive ? "TIM-VX/NPU" : "CPU"))
                  << ")\n";
    }
    return 0;
}
