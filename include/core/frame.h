#pragma once

#include "core/camera.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <memory>
#include <vector>

namespace semantic_vslam {

class Frame {
public:
    using Ptr = std::shared_ptr<Frame>;
    
    Frame(int id, double timestamp, const cv::Mat& image, 
          Camera::Ptr camera);
    
    // Feature extraction
    void extractFeatures(int num_features = 1000);

    //Rectify Image
    void rectifyImage();
    
    // Getters
    int id() const { return id_; }
    double timestamp() const { return timestamp_; }
    const cv::Mat& image() const { return image_; }
    
    Camera::Ptr camera() const { return camera_; }
    
    const std::vector<cv::KeyPoint>& keypoints() const { 
        return keypoints_; 
    }
    const cv::Mat& descriptors() const { 
        return descriptors_; 
    }
    
    // Pose (T_world_camera - 4x4 transformation matrix)
    Eigen::Matrix4d pose() const { return pose_; }
    void setPose(const Eigen::Matrix4d& pose) { pose_ = pose; }
    

    // Visualization
    cv::Mat drawKeypoints() const;

private:
    int id_;
    double timestamp_;
    cv::Mat image_;           // Grayscale image
    Camera::Ptr camera_;
    
    // Features
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;     // ORB descriptors (binary)
    
    // Pose
    Eigen::Matrix4d pose_;    // Default: Identity (world frame)
    
    // ORB detector
    cv::Ptr<cv::ORB> orb_detector_;
};

} // namespace semantic_vslam