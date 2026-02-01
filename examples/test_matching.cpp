#include "core/camera.h"
#include "core/frame.h"
#include "frontend/feature_matcher.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;
using namespace semantic_vslam;

int main(int argc, char** argv) {
    std::cout << "=== Feature Matching Test ===" << std::endl;
    
    // Paths
    std::string data_root = "../data/dataset/sequences/00";
    std::string image_dir = data_root + "/image_0";
    
    if (!fs::exists(image_dir)) {
        std::cerr << "ERROR: Image directory not found!" << std::endl;
        return -1;
    }
    
    // Create camera
    auto camera = std::make_shared<Camera>(
        718.856, 718.856, 607.1928, 185.2157, 1241, 376
    );
    
    // Create feature matcher
    FeatureMatcher matcher(0.75, 50.0);  // ratio=0.75, max_dist=50
    
    // Get image files
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }
    std::sort(image_files.begin(), image_files.end());
    
    std::cout << "\nFound " << image_files.size() << " images" << std::endl;
    std::cout << "Testing matching on consecutive frames" << std::endl;
    std::cout << "Press any key to advance, ESC to quit\n" << std::endl;
    
    // Process pairs of consecutive frames
    int num_pairs = std::min(50, (int)image_files.size() - 1);
    
    Frame::Ptr prev_frame = nullptr;
    
    for (int i = 0; i < num_pairs; i++) {
        // Load current image
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        
        // Create frame
        auto curr_frame = std::make_shared<Frame>(i, i * 0.1, image, camera);
        curr_frame->extractFeatures(1000);
        
        // Match with previous frame
        if (prev_frame) {
            auto matches = matcher.match(prev_frame, curr_frame);
            matcher.printMatchStats(matches);
            
            // Visualize
            cv::Mat img_matches = matcher.visualizeMatches(
                prev_frame, curr_frame, matches
            );
            
            cv::imshow("Feature Matching", img_matches);
            
            int key = cv::waitKey(0);
            if (key == 27) break;  // ESC
        }
        
        prev_frame = curr_frame;
    }
    
    cv::destroyAllWindows();
    
    std::cout << "\n✅ Matching test completed!" << std::endl;
    return 0;
}