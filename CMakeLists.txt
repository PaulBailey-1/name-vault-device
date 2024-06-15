cmake_minimum_required(VERSION 3.18)

project(Name-Vault-Device)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-O2")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_TOOLCHAIN_FILE ./piToolchain.cmake)

set(SRCS
    src/main.cpp
    src/VideoSource.cpp
    src/Display.cpp
    src/Detector.cpp
)

find_package( OpenCV REQUIRED CONFIG)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# libedgetpu
# include_directories( ${CMAKE_SYSROOT}/usr/include/ )
find_library(EDGETPU_LIB libedgetpu.so.1 ) # ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/

# tensorflow-lite
# include_directories( ${CMAKE_SYSROOT}/usr )
# include_directories( ${CMAKE_SYSROOT}/usr/local/include/ )
find_library(TFLITE_LIB libtensorflow-lite.a PATHS /home/paul/tensorflow-2.6.0/tensorflow/lite/tools/make/gen/linux_x86_64/lib ) #${CMAKE_SYSROOT}/usr/local/lib/
find_library(FLATBUFFERS_LIB libflatbuffers.a ) #${CMAKE_SYSROOT}/usr/local/lib/

if (CMAKE_CROSSCOMPILING)

    add_compile_definitions(CROSSCOMPILING=true)
    list(APPEND SRCS src/LibCamera.cpp)
    find_library(LIBCAMERA_LIBRARY libcamera.so REQUIRED)
    find_library(LIBCAMERA_BASE_LIBRARY libcamera-base.so REQUIRED)

    # Flatbuffers
    include_directories( ${CMAKE_SYSROOT}/usr/local/include/ )

else()

    # tensorflow-lite
    include_directories( /home/paul/tensorflow-2.6.0 )

endif()

add_executable(nameVault ${SRCS})
target_link_libraries( nameVault ${OpenCV_LIBS} Threads::Threads ${EDGETPU_LIB} ${TFLITE_LIB} ${FLATBUFFERS_LIB})
target_include_directories( nameVault PUBLIC ./include/)

if (CMAKE_CROSSCOMPILING)
    target_link_libraries( nameVault libcamera.so libcamera-base.so)
    target_include_directories( nameVault PUBLIC ${CMAKE_SYSROOT}/usr/include/libcamera/)
endif()
