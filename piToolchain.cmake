message("Cross compiling for raspberrypi")
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_GENERATOR "Unix Makefiles")

set(TOOLCHAIN_DIR /usr/bin)
set(CMAKE_SYSROOT /home/paul/raspberrypi/sysroot)
link_directories(/usr/aarch64-linux-gnu/lib)
include_directories(/usr/aarch64-linux-gnu/include)

set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_CROSSCOMPILING true)

set(OpenCV_DIR ${CMAKE_SYSROOT}/usr/local/lib/cmake/opencv4/)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)