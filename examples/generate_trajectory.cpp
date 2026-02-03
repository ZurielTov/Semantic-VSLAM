#include "core/camera.h"
#include "core/frame.h"
#include "frontend/feature_matcher.h"
#include "frontend/visual_odometry.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>


namespace fs = std::filesystem;
using namespace semantic_vslam;

// Load KITTI ground truth poses
std::vector<Eigen::Matrix4d> loadGroundTruthPoses(const std::string& filepath) {
    std::vector<Eigen::Matrix4d> poses;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open ground truth file: " << filepath << std::endl;
        return poses;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        
        // KITTI format: 12 values (3x4 matrix [R|t])
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                iss >> pose(i, j);
            }
        }
        
        poses.push_back(pose);
    }
    
    std::cout << "Loaded " << poses.size() << " ground truth poses" << std::endl;
    return poses;
}

int main() {
    std::cout << "=== Trajectory Generation ===" << std::endl;
    
    // Paths
    std::string data_root = "../data/dataset/sequences/00";
    std::string image_dir = data_root + "/image_0";
    std::string gt_file = "../data/dataset/poses/00.txt";
    std::string output_file = "../results/trajectory.txt";
    
    // Load ground truth
    auto gt_poses = loadGroundTruthPoses(gt_file);
    if (gt_poses.empty()) {
        std::cerr << "ERROR: No ground truth loaded!" << std::endl;
        return -1;
    }


    //Create Dist coeff
    cv::Mat D = cv::Mat::zeros(1,5,CV_64F);
    double fx = 718.856;
    double fy = 718.856;
    double cx = 607.1928;
    double cy = 185.2157;

    FeatureMatcher matcher(0.75, 50.0);
    VisualOdometry vo(fx, Eigen::Vector2d(cx, cy));
    
    // Get images
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }
    std::sort(image_files.begin(), image_files.end());

    // Load first image to get size
    cv::Mat first = cv::imread(image_files[0], cv::IMREAD_GRAYSCALE);
    if (first.empty()) {
        std::cerr << "ERROR: Could not read first image: " << image_files[0] << std::endl;
        return -1;
    }

    int width  = first.cols;
    int height = first.rows;

    
    // Create camera
    auto camera = std::make_shared<Camera>(
    fx, fy, cx, cy,
    D,
    width, height
    );
    camera->precomputeUndistortMaps();
    
    //int num_frames = (int)image_files.size();
    int num_frames = std::min(100, (int)image_files.size());
    std::cout << "\nProcessing " << num_frames << " frames..." << std::endl;
    
    // Process frames
    Frame::Ptr prev_frame = nullptr;
    Eigen::Matrix4d accumulated_pose = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> estimated_trajectory;
    std::vector<Eigen::Vector3d> gt_trajectory;
    
    // Initialize: scale first translation to match ground truth
    // (Only for visualization - in real SLAM you don't have this!)
    bool initialized = false;
    double scale = 1.0;
    
    for (int i = 0; i < num_frames; i++) {
        // Load image
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        
        // Create frame
        auto curr_frame = std::make_shared<Frame>(i, i * 0.1, image, camera);

        //curr_frame->rectifyImage();
        curr_frame->extractFeatures(1000);
        
        if (prev_frame) {
            // Match & estimate motion
            auto matches = matcher.match(prev_frame, curr_frame);
            auto motion = vo.estimateMotion(prev_frame, curr_frame, matches);
            
            if (motion.isValid()) {
                // Scale estimation (only for first frame)
                if (!initialized && i == 1) {
                    // Get ground truth translation magnitude
                    Eigen::Vector3d gt_t = gt_poses[1].block<3,1>(0,3) - 
                                           gt_poses[0].block<3,1>(0,3);
                    scale = gt_t.norm();  // Typically ~0.5 meters for KITTI
                    initialized = true;
                    std::cout << "\nInitialized with scale: " << scale << std::endl;
                }
                
                // Apply scale to translation
                Eigen::Matrix4d scaled_motion = motion.toSE3();
                scaled_motion.block<3,1>(0,3) *= scale;
                
                // Accumulate
                accumulated_pose = accumulated_pose * scaled_motion;
            }
        }
        
        // Store trajectory
        Eigen::Vector3d estimated_pos = accumulated_pose.block<3,1>(0,3);
        Eigen::Vector3d gt_pos = gt_poses[i].block<3,1>(0,3);
        
        estimated_trajectory.push_back(estimated_pos);
        gt_trajectory.push_back(gt_pos);
        
        if (i % 10 == 0) {
            std::cout << "Frame " << i << ": "
                     << "Est=(" << estimated_pos.x() << "," << estimated_pos.z() << "), "
                     << "GT=(" << gt_pos.x() << "," << gt_pos.z() << ")" << std::endl;
        }
        
        prev_frame = curr_frame;
    }
    
    // Save trajectory to file
    std::ofstream out_file(output_file);
    out_file << "# Frame X_est Z_est X_gt Z_gt" << std::endl;
    for (size_t i = 0; i < estimated_trajectory.size(); i++) {
        out_file << i << " "
                << estimated_trajectory[i].x() << " "
                << estimated_trajectory[i].z() << " "
                << gt_trajectory[i].x() << " "
                << gt_trajectory[i].z() << std::endl;
    }
    out_file.close();
    
    std::cout << "\n✅ Trajectory saved to: " << output_file << std::endl;
    std::cout << "Use Python script to visualize" << std::endl;
    
    return 0;
}