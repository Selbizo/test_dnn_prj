#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::dnn;

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

int main(int argc, char** argv) {
    // Load model
    std::string weightPath = "/home/selbizo/CV/test_dnn_prj/src/yolov8n.onnx";
    Net net = readNet(weightPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    // Load classes
    getClasses("/home/selbizo/CV/test_dnn_prj/src/object_detection_classes_yolo.txt");
    
    // Read image
    Mat img = imread("/home/selbizo/CV/test_dnn_prj/src/image.jpg");
    
    // Preprocess
    Mat blob;
    Size size(640, 640);
    Scalar mean(0, 0, 0);
    Scalar scale(1.0/255.0, 1.0/255.0, 1.0/255.0);
    blobFromImage(img, blob, 1.0/255.0, size, mean, true, false);
    
    // Forward pass
    net.setInput(blob);
    std::vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    
    // Postprocess
    std::vector<int> keep_classIds;
    std::vector<float> keep_confidences;
    std::vector<Rect2d> keep_boxes;
    yoloPostProcessing(outs, keep_classIds, keep_confidences, keep_boxes,
                      0.5, 0.4, "yolov8", 80);
    
    // Draw results
    std::vector<Rect> boxes;
    for (auto box : keep_boxes) {
        boxes.push_back(Rect(cvFloor(box.x), cvFloor(box.y), 
                           cvFloor(box.width - box.x), cvFloor(box.height - box.y)));
    }
    
    for (size_t idx = 0; idx < boxes.size(); ++idx) {
        Rect box = boxes[idx];
        drawPrediction(keep_classIds[idx], keep_confidences[idx], 
                     box.x, box.y, box.width + box.x, box.height + box.y, img);
    }
    
    imshow("YOLO Object Detection", img);
    waitKey(0);
    
    return 0;
}