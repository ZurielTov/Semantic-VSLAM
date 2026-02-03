#pragma once

#include "core/frame.h"
#include "frontend/feature_matcher.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace semantic_vslam {

class VisualOdometry {
public:
    struct MotionEstimate {
        Eigen::Matrix3d R;          // Rotation matrix (3x3)
        Eigen::Vector3d t;          // Translation vector (normalized, ||t||=1)
        std::vector<int> inliers;   // Indices of inlier matches
        int num_inliers;            // Count of inliers
        double inlier_ratio;        // inliers / total_matches
        
        // Convert to SE(3) transformation matrix (4x4)
        Eigen::Matrix4d toSE3() const {
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = R;
            T.block<3,1>(0,3) = t;
            return T;
        }
        
        bool isValid() const {
            return num_inliers > 50 && inlier_ratio > 0.5;
        }
    };
    
    // Constructor
    VisualOdometry(double focal_length, const Eigen::Vector2d& principal_point);
    
    // Estimate motion between two frames given matches
    MotionEstimate estimateMotion(
        const Frame::Ptr& frame1,
        const Frame::Ptr& frame2,
        const std::vector<FeatureMatcher::Match>& matches
    );
    
    // Print motion statistics
    void printMotionStats(const MotionEstimate& motion) const;

private:
    // Camera intrinsics
    double focal_length_;
    Eigen::Vector2d principal_point_;
    
    // RANSAC parameters
    double ransac_threshold_;    // Reprojection error threshold
    double ransac_confidence_;   // Confidence level
    int ransac_max_iterations_;  // Max iterations
    
    // Convert matches to point correspondences
    void matchesToPoints(
        const Frame::Ptr& frame1,
        const Frame::Ptr& frame2,
        const std::vector<FeatureMatcher::Match>& matches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2
    ) const;
    
    // Convert Eigen to OpenCV and vice versa
    Eigen::Matrix3d cvMatToEigen3d(const cv::Mat& mat) const;
    Eigen::Vector3d cvMatToEigen3dVec(const cv::Mat& mat) const;
};

} // namespace semantic_vslam