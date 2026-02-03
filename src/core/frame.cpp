#include "core/frame.h"
#include <iostream>

namespace semantic_vslam {

Frame::Frame(int id, double timestamp, const cv::Mat& image,
             Camera::Ptr camera)
    : id_(id), timestamp_(timestamp), camera_(camera),
      pose_(Eigen::Matrix4d::Identity()) {
    
    // Convert to grayscale if needed
    if (image.channels() == 3) {
        cv::cvtColor(image, image_, cv::COLOR_BGR2GRAY);
    } else {
        image_ = image.clone();
    }
    
    // Create ORB detector
    orb_detector_ = cv::ORB::create();
    
    std::cout << "Frame " << id_ << " created at t=" 
              << timestamp_ << "s" << std::endl;
}

void Frame::extractFeatures(int num_features) {
    orb_detector_->setMaxFeatures(num_features);
    orb_detector_->detectAndCompute(
        image_, 
        cv::Mat(),  // No mask
        keypoints_, 
        descriptors_
    );
    
    std::cout << "  Frame " << id_ << ": Extracted " 
              << keypoints_.size() << " features" << std::endl;
}

void Frame::rectifyImage() {
    if (!camera_ || image_.empty())
        return;

    // If no distortion or maps not prepared → do nothing
    if (!camera_->hasUndistortMaps())
        return;

    cv::Mat rectified;
    cv::remap(
        image_,
        rectified,
        camera_->map1(),
        camera_->map2(),
        cv::INTER_LINEAR
    );

    image_ = rectified;
}


cv::Mat Frame::drawKeypoints() const {
    cv::Mat img_with_kp;
    cv::drawKeypoints(
        image_, 
        keypoints_, 
        img_with_kp,
        cv::Scalar(0, 255, 0),  // Green
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );
    return img_with_kp;
}

} // namespace semantic_vslam