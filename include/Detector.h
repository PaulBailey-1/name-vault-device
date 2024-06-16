#pragma once

// This file is mostly a compilation of https://github.com/Qengineering/TensorFlow_Lite_SSD_RPi_64-bits
// and https://coral.ai/docs/edgetpu/tflite-cpp/

#include <vector>

#include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/core/ocl.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include <tensorflow/lite/model.h>
#include <edgetpu.h>

struct Detection {
    float x1, y1, x2, y2;
    float score;
    const char* label;
}; 

class Detector {
public:

    Detector() {};
    Detector(std::string modelPath, std::string labelsPath, double confidenceThresh=0.5, bool useTpu=false);
    std::vector<Detection> detect(cv::Mat& src);

private:

    std::unique_ptr<tflite::Interpreter> buildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context);
    bool readFileContents(std::string fileName, std::vector<std::string>& lines);

    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    std::shared_ptr<edgetpu::EdgeTpuContext> _edgetpu_context;
    std::vector<std::string> _labels;

    double _confidenceThresh;

};