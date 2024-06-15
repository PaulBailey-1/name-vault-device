#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Detector.h"

class Display {
public:

    Display();
    bool show(cv::Mat& frame, std::vector<Detection>& detections, double fps);

private:

    void visualize(cv::Mat& input, std::vector<Detection>& detections, double fps);

};