
// From https://github.com/edward-ardu/libcamera-cpp-demo/tree/main

#include <atomic>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <limits.h>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <mutex>

#include <libcamera/controls.h>
#include <libcamera/control_ids.h>
#include <libcamera/property_ids.h>
#include <libcamera/libcamera.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>
#include <libcamera/formats.h>
#include <libcamera/transform.h>

// using namespace libcamera;

typedef struct {
    uint8_t *imageData;
    uint32_t size;
    uint64_t request;
} LibcameraOutData;

class LibCamera {
    public:
        LibCamera(){};
        ~LibCamera(){};
        
        int initCamera();
        void configureStill(int width, int height, libcamera::PixelFormat format, int buffercount, int rotation);
        int startCamera();
        int resetCamera(int width, int height, libcamera::PixelFormat format, int buffercount, int rotation);
        bool readFrame(LibcameraOutData *frameData);
        void returnFrameBuffer(LibcameraOutData frameData);

        void set(libcamera::ControlList controls);
        void stopCamera();
        void closeCamera();

        libcamera::Stream *VideoStream(uint32_t *w, uint32_t *h, uint32_t *stride) const;
        std::string getCameraId();

    private:
        int startCapture();
        int queueRequest(libcamera::Request *request);
        void requestComplete(libcamera::Request *request);
        void processRequest(libcamera::Request *request);

        void StreamDimensions(libcamera::Stream const *stream, uint32_t *w, uint32_t *h, uint32_t *stride) const;

        unsigned int cameraIndex_;
	    uint64_t last_;
        std::unique_ptr<libcamera::CameraManager> cm;
        std::shared_ptr<libcamera::Camera> camera_;
        bool camera_acquired_ = false;
        bool camera_started_ = false;
	    std::unique_ptr<libcamera::CameraConfiguration> config_;
        std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
        std::vector<std::unique_ptr<libcamera::Request>> requests_;
        // std::map<std::string, Stream *> stream_;
        std::map<int, std::pair<void *, unsigned int>> mappedBuffers_;

        std::queue<libcamera::Request *> requestQueue;

        libcamera::ControlList controls_;
        std::mutex control_mutex_;
        std::mutex camera_stop_mutex_;
        std::mutex free_requests_mutex_;

        libcamera::Stream *viewfinder_stream_ = nullptr;
        std::string cameraId;
};