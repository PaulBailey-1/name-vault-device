#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Display {
public:

    Display();
    bool show(cv::Mat& frame, cv::Mat& faces, double fps);

private:

    void visualize(cv::Mat& input, cv::Mat& faces, double fps, int thickness = 2);

};