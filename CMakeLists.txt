cmake_minimum_required(VERSION 3.18)

project(Name-Vault-Device)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-O2")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_TOOLCHAIN_FILE ./piToolchain.cmake)

set(SRCS
    src/main.cpp
)

find_package( OpenCV REQUIRED CONFIG)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (CMAKE_CROSSCOMPILING)
    add_compile_definitions(CROSSCOMPILING=true)
    list(APPEND SRCS src/LibCamera.cpp)
    find_library(LIBCAMERA_LIBRARY libcamera.so REQUIRED)
    find_library(LIBCAMERA_BASE_LIBRARY libcamera-base.so REQUIRED)
endif()

add_executable(nameVault ${SRCS})
target_link_libraries( nameVault ${OpenCV_LIBS} Threads::Threads)
target_include_directories( nameVault PUBLIC ./include/)

if (CMAKE_CROSSCOMPILING)
    target_link_libraries( nameVault libcamera.so libcamera-base.so)
    target_include_directories( nameVault PUBLIC ${CMAKE_SYSROOT}/usr/include/libcamera/)
endif()