
// This is based off https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html

#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <fstream>

#include "VideoSource.h"
#include "Display.h"
#include "Detector.h"

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv,
        "{help  h                 |            | Print this message}"
        "{video v                 | ../res/face_test.mp4 | Path to the input video}"
        "{model_path m            | ../res/detect_tpu.tflite | Path to the model}"
        "{labels_path l           | ../res/labels.txt | Path to the labels}"
        "{confidence_threshold c  | 0.5        | Filter out detections of score < confidence_threshold}"
        "{use_tpu t               | true       | Use Coral accelerator for object detection}"
        "{display d               | 1          | Display stream [1] or not [0]}"
    );
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string fd_modelPath = "../res/face_detection_yunet_2023mar.onnx";
    std::string fr_modelPath = "../res/face_recognition_sface_2021dec.onnx";

    // float scoreThreshold = parser.get<float>("score_threshold");
    // float nmsThreshold = parser.get<float>("nms_threshold");
    // int topK = parser.get<int>("top_k");
    bool showDisplay = parser.get<int>("display");

    cv::TickMeter tm;
    Display display;

    VideoSource* source;
    Detector detector;
    try {
        #ifdef CROSSCOMPILING
            source = new LibCameraVideoSource(640, 480, 30);
        #else
            source = new FileVideoSource(parser.get<std::string>("video"), 30);
        #endif
        detector = Detector(parser.get<std::string>("model_path"), parser.get<std::string>("labels_path"), parser.get<float>("confidence_threshold"), parser.get<bool>("use_tpu"));
    } catch (std::runtime_error& e) {
        std::cout << "Error - " << e.what() << std::endl;
        return 1;
    }

    // cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(fd_modelPath, "", source->getSize());
    // cv::Ptr<cv::FaceRecognizerSF> faceRecognizer = cv::FaceRecognizerSF::create(fr_modelPath, "");

    int nFrame = 0;
    while (true) {
       
        tm.start();

        cv::Mat frame;
        try {
            source->getFrame(frame);
        } catch (std::runtime_error& e) {
            std::cout << "Error - " << e.what() << std::endl;
            return 1;
        }

        // Inference
        std::vector<Detection> detections = detector.detect(frame);
        
        // if (nFrame % 1 == 0) {
        //     detector->detect(frame, faces);

        //     // Aligning and cropping facial image through the first face of faces detected.
        //     if (faces.rows > 0) {
        //         cv::Mat aligned_face;
        //         faceRecognizer->alignCrop(frame, faces.row(0), aligned_face);

        //         // Run feature extraction with given aligned_face
        //         cv::Mat feature;
        //         faceRecognizer->feature(aligned_face, feature);
        //         feature = feature.clone();
        //     }
        // }

        tm.stop();

        if (showDisplay) {
            if (display.show(frame, detections, tm.getFPS()))
                break;
        } else {
            std::cout << "FPS: " << tm.getFPS() << std::endl;
        }

        ++nFrame;
    }

    std::cout << "Processed " << nFrame << " frames" << std::endl;
    delete source;
    std::cout << "Done." << std::endl;

    return 0;
}