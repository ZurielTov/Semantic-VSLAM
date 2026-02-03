#include "core/camera.h"
#include "core/frame.h"
#include "frontend/feature_matcher.h"
#include "frontend/visual_odometry.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;
using namespace semantic_vslam;

int main() {
    std::cout << "=== Visual Odometry Test ===" << std::endl;
    
    // Paths
    std::string data_root = "../data/dataset/sequences/00";
    std::string image_dir = data_root + "/image_0";
    
    
    //Create Dist coeff
    cv::Mat D = (cv::Mat_<double>(1,5) <<
    -3.791375e-01,  // k1
     2.148119e-01,  // k2
     1.227094e-03,  // p1s
     2.343833e-03,  // p2
    -7.910379e-02   // k3
    );
    
    // Create matcher and visual odometry
    FeatureMatcher matcher(0.75, 50.0);
    VisualOdometry vo(981.2178, Eigen::Vector2d(690.0, 247.1364));
    
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
    981.2178, 975.8994, 690.0, 247.1364,
    D,
    width, height
    );
    camera->precomputeUndistortMaps();

    
    std::cout << "\nProcessing first 10 frame pairs..." << std::endl;
    std::cout << "Press any key to continue, ESC to quit\n" << std::endl;
    
    Frame::Ptr prev_frame = nullptr;
    Eigen::Matrix4d accumulated_pose = Eigen::Matrix4d::Identity();
    
    for (int i = 0; i < std::min(10, (int)image_files.size()); i++) {
        // Load image
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        
        // Create frame
        auto curr_frame = std::make_shared<Frame>(i, i * 0.1, image, camera);

        curr_frame->rectifyImage();
        curr_frame->extractFeatures(1000);
        
        if (prev_frame) {
            // Match features
            auto matches = matcher.match(prev_frame, curr_frame);
            
            // Estimate motion
            auto motion = vo.estimateMotion(prev_frame, curr_frame, matches);
            vo.printMotionStats(motion);
            
            if (motion.isValid()) {
                // Accumulate pose
                accumulated_pose = accumulated_pose * motion.toSE3();
                
                // Extract position
                Eigen::Vector3d position = accumulated_pose.block<3,1>(0,3);
                std::cout << "\nCurrent position: (" 
                         << position.x() << ", "
                         << position.y() << ", "
                         << position.z() << ")" << std::endl;
            }
            
            // Visualize matches with inliers highlighted
            cv::Mat img_matches = matcher.visualizeMatches(
                prev_frame, curr_frame, matches
            );
            cv::imshow("Matches", img_matches);
            
            int key = cv::waitKey(0);
            if (key == 27) break;
        }
        
        prev_frame = curr_frame;
    }
    
    cv::destroyAllWindows();
    
    std::cout << "\n✅ Visual odometry test completed!" << std::endl;
    return 0;
}