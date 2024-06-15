
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
    _cam.configureStream(width, height, libcamera::formats::RGB888, 1, 0);

    std::cout << "Starting camera\n";
    _cam.startCamera();

    _cam.VideoStream(&_frameWidth, &_frameHeight, &_stride);
    std::cout << "Starting video stream of " << _frameWidth << "X" << _frameHeight << " and stride " << _stride << "\n";
}

LibCameraVideoSource::~LibCameraVideoSource() {
    _cam.stopCamera();
    _cam.closeCamera();
}

void LibCameraVideoSource::getFrame(cv::Mat& frame) {
    int count = 0;
    while (!_cam.readFrame(&_frameData)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        count++;
    }
    std::cout << "Slept " << count << std::endl;
    // _frameData.imageData is non-modifiable, remove clone if no display
    frame = cv::Mat(_frameHeight, _frameWidth, CV_8UC3, _frameData.imageData, _stride).clone();
    _cam.returnFrameBuffer(_frameData);
}

void LibCameraVideoSource::returnFrame() {
    _cam.returnFrameBuffer(_frameData);
}

#endif

FileVideoSource::FileVideoSource(std::string path, int frameRate) {
    _frameRate = frameRate;
    _cap.open(path);
    if (_cap.isOpened()) {
        _frameWidth = _cap.get(cv::CAP_PROP_FRAME_WIDTH);
        _frameHeight = _cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Starting video stream from" << path << "of " << _frameWidth << " X " << _frameHeight << "\n";
    } else {
        throw std::runtime_error("FileVideoSource:Could not open " + path);
    }
}

void FileVideoSource::getFrame(cv::Mat& frame) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / _frameRate));
    if (!_cap.read(frame)) {
        throw std::runtime_error("FileVideoSource: Can't grab frame");
    }
}
