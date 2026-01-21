cmake_minimum_required(VERSION 3.10)
project(SemanticVSLAM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

# Find packages
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Core library
add_library(vslam_core SHARED
    src/core/camera.cpp
    src/core/frame.cpp
)

target_link_libraries(vslam_core
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Example: Run KITTI
add_executable(run_kitti examples/run_kitti.cpp)
target_link_libraries(run_kitti vslam_core)

# Test executable
add_executable(test_camera test/test_camera.cpp)
target_link_libraries(test_camera vslam_core)

message(STATUS "Build configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")