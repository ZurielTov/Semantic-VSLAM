#include "core/camera.h"
#include <iostream>

namespace semantic_vslam {

Camera::Camera(double fx, double fy, double cx, double cy, const cv::Mat& dist,
               int width, int height)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), dist_(dist.clone()),
      width_(width), height_(height) {
    
    std::cout << "Camera created: " 
              << width << "x" << height 
              << ", f=(" << fx << "," << fy << ")"
              << ", c=(" << cx << "," << cy << ")"
              << std::endl;
}

Eigen::Vector2d Camera::project(const Eigen::Vector3d& p) const {
    // Pinhole projection: u = fx * X/Z + cx
    return Eigen::Vector2d(
        fx_ * p.x() / p.z() + cx_,
        fy_ * p.y() / p.z() + cy_
    );
}

Eigen::Vector3d Camera::backproject(const Eigen::Vector2d& px) const {
    // Normalized ray: direction from camera center through pixel
    return Eigen::Vector3d(
        (px.x() - cx_) / fx_,
        (px.y() - cy_) / fy_,
        1.0
    ).normalized();
}

bool Camera::isInImage(const Eigen::Vector2d& pixel, int border) const {
    return pixel.x() >= border && 
           pixel.x() < width_ - border &&
           pixel.y() >= border && 
           pixel.y() < height_ - border;
}



cv::Mat Camera::Kcv() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx_;
    K.at<double>(1, 1) = fy_;
    K.at<double>(0, 2) = cx_;
    K.at<double>(1, 2) = cy_;
    return K;
}

void Camera::precomputeUndistortMaps(){
    if (dist_.empty() || cv::countNonZero(dist_.reshape(1)) == 0) {
            maps_ready_ = false;
            return;
        }

        cv::Mat K = Kcv();
        cv::initUndistortRectifyMap(
            K, dist_, cv::Mat(), K,
            cv::Size(width_, height_),
            CV_32FC1,
            map1_, map2_
        );
        maps_ready_ = true;
}

} // namespace semantic_vslam