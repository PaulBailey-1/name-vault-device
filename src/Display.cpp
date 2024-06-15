
#include <iostream>
#include <stdio.h>

#include "Display.h"

Display::Display() {}

bool Display::show(cv::Mat& frame, std::vector<Detection>& detections, double fps) {
    // Draw results on the input image
    visualize(frame, detections, fps);

    // Visualize results
    cv::imshow("Stream", frame);

    int key = cv::waitKey(1);
    
    if (key > 0) return true;
    return false;
}

void Display::visualize(cv::Mat& input, std::vector<Detection>& detections, double fps) {
    std::string fpsString = cv::format("FPS : %.2f", (float)fps);
    for (Detection d : detections) {
        // Print results
        printf("Detection of %s, score: %f\n", d.label, d.score);

        // Draw bounding box
        cv::Rect rec((int)d.x1, (int)d.y1, (int)(d.x2 - d.x1), (int)(d.y2 - d.y1));
        rectangle(input, rec, cv::Scalar(0, 0, 255), 2);
        // Draw label
        putText(input, cv::format("%s", d.label), cv::Point2i(d.x1, d.y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

    }
    putText(input, fpsString, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
}