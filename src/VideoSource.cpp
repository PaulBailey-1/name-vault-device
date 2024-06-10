
#include <iostream>
#include <chrono>
#include <thread>
#include <stdexcept>

#include "VideoSource.h"

cv::Size VideoSource::getSize() {
    return cv::Size(_frameWidth, _frameHeight);
}

#ifdef CROSSCOMPILING

LibCameraVideoSource::LibCameraVideoSource(int width, int height) {
    if (_cam.initCamera()) {
        throw std::runtime_error("LibCameraVideoSource:Could not initialize libcamera");
    }
    _cam.configureStill(width, height, libcamera::formats::RGB888, 1, 0);

    std::cout << "Starting camera\n";
    _cam.startCamera();

    _cam.VideoStream(&_frameWidth, &_frameHeight, &_stride);
    std::cout << "Starting video stream of " << _frameWidth << "X" << _frameHeight << " and stride " << _stride << "\n";
}

LibCameraVideoSource::~LibCameraVideoSource() {
    _cam.stopCamera();
    _cam.closeCamera();
}

std::unique_ptr<cv::Mat> LibCameraVideoSource::getFrame() {
    while (!_cam.readFrame(&_frameData)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::unique_ptr<cv::Mat>(new cv::Mat(_frameHeight, _frameWidth, CV_8UC3, _frameData.imageData, _stride));
}

void LibCameraVideoSource::returnFrame() {
    _cam.returnFrameBuffer(_frameData);
}

#endif

FileVideoSource::FileVideoSource(std::string path) {
    _cap.open(path);
    if (_cap.isOpened()) {
        _frameWidth = _cap.get(cv::CAP_PROP_FRAME_WIDTH);
        _frameHeight = _cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Starting video stream from" << path << "of " << _frameWidth << " X " << _frameHeight << "\n";
    } else {
        throw std::runtime_error("FileVideoSource:Could not open " + path);
    }
}

std::unique_ptr<cv::Mat> FileVideoSource::getFrame() {
    std::unique_ptr<cv::Mat> frame = std::unique_ptr<cv::Mat>(new cv::Mat());
    if (!_cap.read(*frame)) {
        throw std::runtime_error("FileVideoSource: Can't grab frame");
    }
    return frame;
}
