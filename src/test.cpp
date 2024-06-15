#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include <cmath>

#include <cstdio>
#include <memory>
#include "tensorflow/lite/optional_debug_tools.h"
#include "edgetpu.h"

using namespace cv;
using namespace std;

const size_t width = 300;
const size_t height = 300;

std::vector<std::string> Labels;
std::unique_ptr<tflite::Interpreter> interpreter;

static bool getFileContent(std::string fileName)
{

	// Open the File
	std::ifstream in(fileName.c_str());
	// Check if object is valid
	if(!in.is_open()) return false;

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size()>0) Labels.push_back(str);
	}
	// Close The File
	in.close();
	return true;
}

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context ) {
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
    }
    // Bind given context with interpreter.
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
    interpreter->SetNumThreads(1);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
    }
    return interpreter;
}

void detect_from_video(Mat &src)
{
    Mat image;
    int cam_width =src.cols;
    int cam_height=src.rows;

    // copy image to input as input tensor
    cv::resize(src, image, Size(300,300));
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      //quad core

//        cout << "tensors size: " << interpreter->tensors_size() << "\n";
//        cout << "nodes size: " << interpreter->nodes_size() << "\n";
//        cout << "inputs: " << interpreter->inputs().size() << "\n";
//        cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
//        cout << "outputs: " << interpreter->outputs().size() << "\n";

    interpreter->Invoke();      // run your model

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    //there are ALWAYS 10 detections no matter how many objects are detectable
//        cout << "number of detections: " << num_detections << "\n";

    const float confidence_threshold = 0.5;
    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > confidence_threshold){
            int  det_index = (int)detection_classes[i]+1;
            float y1=detection_locations[4*i  ]*cam_height;
            float x1=detection_locations[4*i+1]*cam_width;
            float y2=detection_locations[4*i+2]*cam_height;
            float x2=detection_locations[4*i+3]*cam_width;

            Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            rectangle(src,rec, Scalar(0, 0, 255), 1, 8, 0);
            putText(src, format("%s", Labels[det_index].c_str()), Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 0, 255), 1, 8, 0);
        }
    }
}

int main(int argc,char ** argv)
{
    float f;
    float FPS[16];
    int i, Fcnt=0;
    Mat frame;
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("../res/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite");

    // Build the interpreter
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    interpreter = BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
    // tflite::ops::builtin::BuiltinOpResolver resolver;
    // tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    // interpreter->AllocateTensors();

	// Get the names
	bool result = getFileContent("../res/COCO_labels.txt");
	if(!result)
	{
        cout << "loading labels failed";
        exit(-1);
	}

    VideoCapture cap("../res/face_test.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;
	while(1){
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }

        Tbegin = chrono::steady_clock::now();

        detect_from_video(frame);

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(frame, format("FPS %0.2f", f/16),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));

        //show output
//        cout << "FPS" << f/16 << endl;
        imshow("RPi 4 - 1,9 GHz - 2 Mb RAM", frame);

        char esc = waitKey(5);
        if(esc == 27) break;
    }

  cout << "Closing the camera" << endl;
  destroyAllWindows();
  cout << "Bye!" << endl;

  return 0;
}

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// #include <cstdio>
// #include <memory>
// #include <iostream>

// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"

// #include "edgetpu.h"

// // This is an example that is minimal to read a model
// // from disk and perform inference. There is no data being loaded
// // that is up to you to add as a user.

// // NOTE: Do not add any dependencies to this that cannot be built with
// // the minimal makefile. This example must remain trivial to build with
// // the minimal build tool.

// // Usage: minimal <tflite model>

// #define TFLITE_MINIMAL_CHECK(x)                              \
//   if (!(x)) {                                                \
//     fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
//     exit(1);                                                 \
//   }

// std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context ) {
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
//     std::unique_ptr<tflite::Interpreter> interpreter;
//     if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
//         std::cerr << "Failed to build interpreter." << std::endl;
//     }
//     // Bind given context with interpreter.
//     interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
//     interpreter->SetNumThreads(1);
//     if (interpreter->AllocateTensors() != kTfLiteOk) {
//         std::cerr << "Failed to allocate tensors." << std::endl;
//     }
//     return interpreter;
// }

// int main(int argc, char* argv[]) {

//   // Load model
//   std::string filename = "../res/detect.tflite";
//   std::unique_ptr<tflite::FlatBufferModel> model =
//       tflite::FlatBufferModel::BuildFromFile(filename.c_str());
//   TFLITE_MINIMAL_CHECK(model != nullptr);

//   std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
//   std::unique_ptr<tflite::Interpreter> interpreter = BuildEdgeTpuInterpreter(*model, edgetpu_context.get());

//   // Fill input buffers
//   // TODO(user): Insert code to fill input tensors.
//   // Note: The buffer of the input tensor with index `i` of type T can
//   // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
//   unsigned char* input = interpreter->typed_input_tensor<unsigned char>(0);
//   printf("Input tensor: %p\n", (void*) input);

//   // Run inference
//   TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
//   printf("\n\n=== Post-invoke Interpreter State ===\n");
// //   tflite::PrintInterpreterState(interpreter.get());

//   // Read output buffers
//   // TODO(user): Insert getting data out code.
//   // Note: The buffer of the output tensor with index `i` of type T can
//   // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

//   return 0;
// }