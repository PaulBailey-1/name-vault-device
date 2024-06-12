#pragma once

#include <string>
#include <memory>

#include <opencv2/videoio.hpp>

#ifdef CROSSCOMPILING
#include "LibCamera.h"
#endif

class VideoSource {
public:

    virtual ~VideoSource() {}
    cv::Size getSize();
    virtual std::unique_ptr<cv::Mat> getFrame() = 0;
    virtual void returnFrame() {}

protected:

    uint32_t _frameWidth;
    uint32_t _frameHeight;

};

#ifdef CROSSCOMPILING

class LibCameraVideoSource : public VideoSource {
public:

    LibCameraVideoSource(int width, int height);
    ~LibCameraVideoSource();
    std::unique_ptr<cv::Mat> getFrame() override;
    void returnFrame() override;

private:

    uint32_t _stride;

    LibCamera _cam;
    LibcameraOutData _frameData;

};

#endif

class FileVideoSource : public VideoSource {
public:

    FileVideoSource(std::string path, int frameRate);
    std::unique_ptr<cv::Mat> getFrame() override;

private:

    cv::VideoCapture _cap;
    int _frameRate;

};