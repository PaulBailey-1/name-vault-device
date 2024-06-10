#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <fstream>

#ifdef CROSSCOMPILING
    #include "LibCamera.h"
#endif

void visualize(cv::Mat& input, int frame, cv::Mat& faces, double fps, int thickness = 2)
{
    std::string fpsString = cv::format("FPS : %.2f", (float)fps);
    if (frame >= 0)
        std::cout << "Frame " << frame << ", ";
    std::cout << "FPS: " << fpsString << std::endl;
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        std::cout << "Face " << i
             << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
             << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
             << "score: " << cv::format("%.2f", faces.at<float>(i, 14))
             << std::endl;

        // Draw bounding box
        rectangle(input, cv::Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), cv::Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(input, cv::Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, cv::Scalar(255, 0, 0), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, cv::Scalar(0, 0, 255), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, cv::Scalar(0, 255, 0), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, cv::Scalar(255, 0, 255), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, cv::Scalar(0, 255, 255), thickness);
    }
    putText(input, fpsString, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
}

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv,
        "{help  h           |            | Print this message}"
        "{video v           | ./res/face_test.mp4 | Path to the input video}"
        "{score_threshold   | 0.9        | Filter out faces of score < score_threshold}"
        "{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold}"
        "{top_k             | 5000       | Keep top_k bounding boxes before NMS}"
    );
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    uint32_t frameWidth = 640;
    uint32_t frameHeight = 480;
    std::string fd_modelPath = "./res/face_detection_yunet_2023mar.onnx";
    std::string fr_modelPath = "./res/face_recognition_sface_2021dec.onnx";

    float scoreThreshold = parser.get<float>("score_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");

    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(fd_modelPath, "", cv::Size(frameHeight, frameWidth), scoreThreshold, nmsThreshold, topK);

    cv::TickMeter tm;

    #ifdef CROSSCOMPILING
        LibCamera cam;
        if (cam.initCamera()) {
            std::cout << "Could not initialize libcamera\n";
            return 1;
        }
        cam.configureStill(frameWidth, frameHeight, libcamera::formats::RGB888, 1, 0);
        LibcameraOutData frameData;

        std::cout << "Starting camera\n";
        cam.startCamera();
        uint32_t stride;
        cam.VideoStream(&frameWidth, &frameHeight, &stride);
        std::cout << "Starting video stream of " << frameWidth << "X" << frameHeight << " and stride " << stride << "\n";
    #else
        cv::VideoCapture cap;
        std::string inputPath = parser.get<std::string>("video");
        cap.open(inputPath);
        if (cap.isOpened()) {
            frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            std::cout << "Starting video stream from" << inputPath << "of " << frameWidth << " X " << frameHeight << std::endl;
        } else {
            std::cout << "Could not open " << inputPath << "\n";
            return 1;
        }
    #endif

    detector->setInputSize(cv::Size(frameWidth, frameHeight));

    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer = cv::FaceRecognizerSF::create(fr_modelPath, "");
    std::ofstream outputFile("output.csv");

    int nFrame = 0;
    while (true) {
        #ifdef CROSSCOMPILING
            if (!cam.readFrame(&frameData)) {
                continue;
            }
            cv::Mat frame(frameHeight, frameWidth, CV_8UC3, frameData.imageData, stride);
        #else
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::cerr << "Can't grab frame! Stop\n";
                break;
            }
        #endif

        // Inference
        cv::Mat faces;
        tm.start();
        
        detector->detect(frame, faces);

        // Aligning and cropping facial image through the first face of faces detected.
        if (faces.rows > 0) {
            cv::Mat aligned_face;
            faceRecognizer->alignCrop(frame, faces.row(0), aligned_face);

            // Run feature extraction with given aligned_face
            cv::Mat feature;
            faceRecognizer->feature(aligned_face, feature);
            feature = feature.clone();
        }

        tm.stop();

        cv::Mat result = frame.clone();
        // Draw results on the input image
        visualize(result, nFrame, faces, tm.getFPS());

        // Visualize results
        cv::imshow("Live", result);

        int delay = tm.getFPS() > 60 ? 30 : 1;
        int key = cv::waitKey(delay);

        if (key > 0)
            break;

        ++nFrame;
        #ifdef CROSSCOMPILING
            cam.returnFrameBuffer(frameData);
        #endif
    }

    std::cout << "Processed " << nFrame << " frames" << std::endl;

    #ifdef CROSSCOMPILING
        cam.stopCamera();
        cam.closeCamera();
    #endif
    std::cout << "Done." << std::endl;

    return 0;
}