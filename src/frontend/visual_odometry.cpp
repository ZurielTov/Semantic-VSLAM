#include "frontend/visual_odometry.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <Eigen/Geometry>

namespace semantic_vslam {

VisualOdometry::VisualOdometry(
    double focal_length,
    const Eigen::Vector2d& principal_point)
    : focal_length_(focal_length),
      principal_point_(principal_point),
      ransac_threshold_(1.0),      // 1 pixel reprojection error
      ransac_confidence_(0.99),    // 99% confidence
      ransac_max_iterations_(1000) {
    
    std::cout << "VisualOdometry created:" << std::endl;
    std::cout << "  Focal length: " << focal_length_ << std::endl;
    std::cout << "  Principal point: (" << principal_point_.x() 
              << ", " << principal_point_.y() << ")" << std::endl;
    std::cout << "  RANSAC threshold: " << ransac_threshold_ << " pixels" << std::endl;
}

void VisualOdometry::matchesToPoints(
    const Frame::Ptr& frame1,
    const Frame::Ptr& frame2,
    const std::vector<FeatureMatcher::Match>& matches,
    std::vector<cv::Point2f>& points1,
    std::vector<cv::Point2f>& points2) const {
    
    points1.clear();
    points2.clear();
    points1.reserve(matches.size());
    points2.reserve(matches.size());
    
    const auto& kp1 = frame1->keypoints();
    const auto& kp2 = frame2->keypoints();
    
    for (const auto& match : matches) {
        // Get pixel coordinates from keypoints
        points1.push_back(kp1[match.idx1].pt);
        points2.push_back(kp2[match.idx2].pt);
    }
}

VisualOdometry::MotionEstimate VisualOdometry::estimateMotion(
    const Frame::Ptr& frame1,
    const Frame::Ptr& frame2,
    const std::vector<FeatureMatcher::Match>& matches) {
    
    MotionEstimate motion;
    
    if (matches.size() < 8) {
        std::cerr << "ERROR: Not enough matches (" << matches.size() 
                  << ") for motion estimation (need at least 8)" << std::endl;
        motion.num_inliers = 0;
        motion.inlier_ratio = 0.0;
        return motion;
    }
    
    // Step 1: Convert matches to point correspondences
    std::vector<cv::Point2f> points1, points2;
    matchesToPoints(frame1, frame2, matches, points1, points2);
    
    std::cout << "\n--- Motion Estimation: Frame " << frame1->id() 
              << " -> " << frame2->id() << " ---" << std::endl;
    std::cout << "Input: " << matches.size() << " matches" << std::endl;
    
    // Step 2: Compute Essential Matrix using 5-point algorithm + RANSAC
    cv::Point2d pp(principal_point_(0), principal_point_(1)); 


    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(
        points1,                    // Points in frame1
        points2,                    // Points in frame2
        focal_length_,              // Focal length
        pp,           // Principal point (cx, cy)
        cv::RANSAC,                 // RANSAC method
        ransac_confidence_,         // 99% confidence
        ransac_threshold_,          // 1.0 pixel threshold
        inlier_mask                 // Output: inlier mask
    );
    
    if (E.empty()) {
        std::cerr << "ERROR: Essential matrix estimation failed!" << std::endl;
        motion.num_inliers = 0;
        motion.inlier_ratio = 0.0;
        return motion;
    }
    
    // Count inliers
    motion.num_inliers = cv::countNonZero(inlier_mask);
    motion.inlier_ratio = (double)motion.num_inliers / matches.size();
    
    std::cout << "RANSAC: " << motion.num_inliers << " / " << matches.size() 
              << " inliers (" << (motion.inlier_ratio * 100) << "%)" << std::endl;
    
    // Store inlier indices
    for (int i = 0; i < inlier_mask.rows; i++) {
        if (inlier_mask.at<uchar>(i)) {
            motion.inliers.push_back(i);
        }
    }
    
    // Step 3: Recover pose (R, t) from Essential Matrix
    cv::Mat R_cv, t_cv;


    int inliers_recovered = cv::recoverPose(
        E,                      // Essential matrix
        points1,                // Points in frame1
        points2,                // Points in frame2
        R_cv,                   // Output: Rotation
        t_cv,                   // Output: Translation (normalized)
        focal_length_,          // Focal length
        pp,       // Principal point
        inlier_mask             // Inlier mask (refined)
    );
    
    std::cout << "Pose recovery: " << inliers_recovered << " inliers validated" << std::endl;
    
    // Step 4: Convert OpenCV matrices to Eigen
    Eigen::Matrix3d R_cam1_cam0 = cvMatToEigen3d(R_cv);
    Eigen::Vector3d t_cam1_cam0 = cvMatToEigen3dVec(t_cv);

    motion.R = R_cam1_cam0.transpose();
    motion.t = -(motion.R * t_cam1_cam0);

    // Normalize translation (should already be ~1, but make sure)
    motion.t.normalize();
    
    // Verify rotation matrix properties
    double det = motion.R.determinant();
    if (std::abs(det - 1.0) > 0.01) {
        std::cerr << "WARNING: Rotation matrix determinant = " << det 
                  << " (should be 1.0)" << std::endl;
    }
    
    return motion;
}

void VisualOdometry::printMotionStats(const MotionEstimate& motion) const {
    std::cout << "\n=== Motion Estimate ===" << std::endl;
    
    if (!motion.isValid()) {
        std::cout << "⚠️  MOTION ESTIMATE IS INVALID!" << std::endl;
        std::cout << "    Inliers: " << motion.num_inliers << " (need >50)" << std::endl;
        std::cout << "    Ratio: " << (motion.inlier_ratio * 100) 
                  << "% (need >50%)" << std::endl;
        return;
    }
    
    std::cout << "✅ Valid motion estimate" << std::endl;
    std::cout << "Inliers: " << motion.num_inliers 
              << " (" << (motion.inlier_ratio * 100) << "%)" << std::endl;
    
    // Rotation analysis
    // Convert rotation matrix to axis-angle representation
    Eigen::AngleAxisd angle_axis(motion.R);
    double angle_deg = angle_axis.angle() * 180.0 / M_PI;
    Eigen::Vector3d axis = angle_axis.axis();
    
    std::cout << "\nRotation:" << std::endl;
    std::cout << "  Angle: " << angle_deg << " degrees" << std::endl;
    std::cout << "  Axis: (" << axis.x() << ", " << axis.y() 
              << ", " << axis.z() << ")" << std::endl;
    
    // Translation (normalized)
    std::cout << "\nTranslation (normalized):" << std::endl;
    std::cout << "  Direction: (" << motion.t.x() << ", " << motion.t.y() 
              << ", " << motion.t.z() << ")" << std::endl;
    std::cout << "  Norm: " << motion.t.norm() << " (should be 1.0)" << std::endl;
    
    // Interpret motion
    if (std::abs(motion.t.z()) > 0.8) {
        std::cout << "  → Mostly FORWARD/BACKWARD motion" << std::endl;
    }
    if (std::abs(motion.t.x()) > 0.5) {
        std::cout << "  → Significant LEFT/RIGHT motion" << std::endl;
    }
    if (angle_deg > 5.0) {
        std::cout << "  → Significant rotation (" << angle_deg << "°)" << std::endl;
    }
}

Eigen::Matrix3d VisualOdometry::cvMatToEigen3d(const cv::Mat& mat) const {
    Eigen::Matrix3d eigen_mat;
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            eigen_mat(i, j) = mat.at<double>(i, j);
        }
    }
    
    return eigen_mat;
}

Eigen::Vector3d VisualOdometry::cvMatToEigen3dVec(const cv::Mat& mat) const {
    return Eigen::Vector3d(
        mat.at<double>(0, 0),
        mat.at<double>(1, 0),
        mat.at<double>(2, 0)
    );
}

} // namespace semantic_vslam