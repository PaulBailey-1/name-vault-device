#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <fstream>

#include "VideoSource.h"
#include "Display.h"

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv,
        "{help  h           |            | Print this message}"
        "{video v           | ./res/face_test.mp4 | Path to the input video}"
        "{score_threshold   | 0.9        | Filter out faces of score < score_threshold}"
        "{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold}"
        "{top_k             | 5000       | Keep top_k bounding boxes before NMS}"
        "{display d         | 1          | Display stream [1] or not [0]}"
    );
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string fd_modelPath = "./res/face_detection_yunet_2023mar.onnx";
    std::string fr_modelPath = "./res/face_recognition_sface_2021dec.onnx";

    float scoreThreshold = parser.get<float>("score_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");
    bool showDisplay = parser.get<int>("display");

    cv::TickMeter tm;
    Display display;

    VideoSource* source;
    try {
        #ifdef CROSSCOMPILING
            source = new LibCameraVideoSource(640, 480);
        #else
            source = new FileVideoSource(parser.get<std::string>("video"));
        #endif
    } catch (std::runtime_error& e) {
        std::cout << "Error - " << e.what() << std::endl;
        return 1;
    }

    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(fd_modelPath, "", source->getSize(), scoreThreshold, nmsThreshold, topK);
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer = cv::FaceRecognizerSF::create(fr_modelPath, "");

    int nFrame = 0;
    while (true) {
       
        std::unique_ptr<cv::Mat> frame;
        try {
            frame = source->getFrame();
        } catch (std::runtime_error& e) {
            std::cout << "Error - " << e.what() << std::endl;
            return 1;
        }

        // Inference
        cv::Mat faces;
        tm.start();
        
        detector->detect(*frame, faces);

        // Aligning and cropping facial image through the first face of faces detected.
        if (faces.rows > 0) {
            cv::Mat aligned_face;
            faceRecognizer->alignCrop(*frame, faces.row(0), aligned_face);

            // Run feature extraction with given aligned_face
            cv::Mat feature;
            faceRecognizer->feature(aligned_face, feature);
            feature = feature.clone();
        }

        tm.stop();

        if (showDisplay) { 
            if (display.show(*frame, faces, tm.getFPS()))
                break;
        } else {
            std::cout << "FPS: " << tm.getFPS() << std::endl;
        }

        source->returnFrame();
        ++nFrame;
    }

    std::cout << "Processed " << nFrame << " frames" << std::endl;
    delete source;
    std::cout << "Done." << std::endl;

    return 0;
}