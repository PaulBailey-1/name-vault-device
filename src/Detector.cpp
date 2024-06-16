
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include "Detector.h"

Detector::Detector(std::string modelPath, std::string labelsPath, double confidenceThresh, bool useTpu) {

    _confidenceThresh = confidenceThresh;

    printf("Creating detector with %s\n", modelPath.c_str());

    // Load model
    _model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

    // Build the interpreter
    if (useTpu) {
        _edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
        printf("Opened Coral TPU Accelerator\n");
        _interpreter = buildEdgeTpuInterpreter(*_model, _edgetpu_context.get());
    } else {
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*_model.get(), resolver)(&_interpreter);
        if (tflite::InterpreterBuilder(*_model.get(), resolver)(&_interpreter) != kTfLiteOk) {
            throw std::runtime_error("Detector::Detector - Failed to build interpreter");
        }
        if (_interpreter->AllocateTensors() != kTfLiteOk) {
            throw std::runtime_error("Detector::Detector - Failed to allocate tensors");
        }
        _interpreter->SetNumThreads(3);
    }
    _interpreter->SetAllowFp16PrecisionForFp32(true);

	// Read labels file
	if(!readFileContents(labelsPath, _labels)) {
        throw std::runtime_error("Detector::Detector - Could not load labels file");
	}
}

std::unique_ptr<tflite::Interpreter> Detector::buildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context) {
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
        throw std::runtime_error("Detector::buildEdgeTpuInterpreter - Failed to build interpreter");
    }
    // Bind given context with interpreter.
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
    interpreter->SetNumThreads(3);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Detector::buildEdgeTpuInterpreter - Failed to allocate tensors");
    }
    return interpreter;
}

bool Detector::readFileContents(std::string fileName, std::vector<std::string>& lines) {
	std::ifstream in(fileName.c_str());
	if(!in.is_open()) return false;

	std::string str;
	while (std::getline(in, str)) {
		if(str.size() > 0)
            lines.push_back(str);
	}
	in.close();
	return true;
}

std::vector<Detection> Detector::detect(cv::Mat& src) {

    cv::Mat image;
    int cam_width = src.cols;
    int cam_height = src.rows;

    // copy image to input as input tensor
    cv::resize(src, image, cv::Size(300,300));
    memcpy(_interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

//        cout << "tensors size: " << _interpreter->tensors_size() << "\n";
//        cout << "nodes size: " << _interpreter->nodes_size() << "\n";
//        cout << "inputs: " << _interpreter->inputs().size() << "\n";
//        cout << "input(0) name: " << _interpreter->GetInputName(0) << "\n";
//        cout << "outputs: " << _interpreter->outputs().size() << "\n";

    _interpreter->Invoke(); // run the model

    const float* detection_locations = _interpreter->tensor(_interpreter->outputs()[0])->data.f;
    const float* detection_classes=_interpreter->tensor(_interpreter->outputs()[1])->data.f;
    const float* detection_scores = _interpreter->tensor(_interpreter->outputs()[2])->data.f;
    const int num_detections = *_interpreter->tensor(_interpreter->outputs()[3])->data.f;

    //there are ALWAYS 10 detections no matter how many objects are detectable

    std::vector<Detection> detections;
    for (int i = 0; i < num_detections; i++) {
        if (detection_scores[i] > _confidenceThresh){
            int det_index = (int) detection_classes[i] + 1;
            Detection d;
            d.y1 = detection_locations[4*i] * cam_height;
            d.x1 = detection_locations[4*i+1] * cam_width;
            d.y2 = detection_locations[4*i+2] * cam_height;
            d.x2 = detection_locations[4*i+3] * cam_width;

            d.score = detection_scores[i];
            d.label = _labels[det_index].c_str();
            detections.push_back(d);
        }
    }
    return detections;
}