#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <memory>

namespace semantic_vslam {

class Camera {
public:
    using Ptr = std::shared_ptr<Camera>;
    
    Camera(double fx, double fy, double cx, double cy, const cv::Mat &,
           int width, int height);
    
    // Project 3D point (in camera frame) to pixel
    Eigen::Vector2d project(const Eigen::Vector3d& point_3d) const;
    
    // Backproject pixel to normalized ray (direction only)
    Eigen::Vector3d backproject(const Eigen::Vector2d& pixel) const;
    
    // Check if pixel is in image bounds
    bool isInImage(const Eigen::Vector2d& pixel, int border = 0) const;

    //Undistort maps
    void precomputeUndistortMaps();
    
    // Getters
    double fx() const { return fx_; }
    double fy() const { return fy_; }
    double cx() const { return cx_; }
    double cy() const { return cy_; }
    int width() const { return width_; }
    int height() const { return height_; }
    //get dist coeff
    const cv::Mat& distCoeffs() const { return dist_; }

    //Used for rectification
    bool hasUndistortMaps() const { return maps_ready_; }
    const cv::Mat& map1() const { return map1_; }
    const cv::Mat& map2() const { return map2_; }

    cv::Mat Kcv() const;  // Intrinsic matrix 3x3

private:
    double fx_, fy_;      // Focal length
    double cx_, cy_;      // Principal point
    cv::Mat dist_;
    int width_, height_;  // Image dimensions
    

    //Cached rectification maps
    cv::Mat map1_,map2_;
    bool maps_ready_;
};

} // namespace semantic_vslam