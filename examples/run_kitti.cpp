#include "core/camera.h"
#include "core/frame.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;
using namespace semantic_vslam;

int main(int argc, char** argv) {
    std::cout << "=== Semantic VSLAM - KITTI Test ===" << std::endl;
    
    // Paths
    std::string data_root = "../data/dataset/sequences/00";
    std::string image_dir = data_root + "/image_0";
    
    if (!fs::exists(image_dir)) {
        std::cerr << "ERROR: Image directory not found: " 
                  << image_dir << std::endl;
        std::cerr << "Did you download KITTI dataset?" << std::endl;
        return -1;
    }

    //Create Dist coeff
    cv::Mat D = (cv::Mat_<double>(1,5) <<
    -3.791375e-01,  // k1
     2.148119e-01,  // k2
     1.227094e-03,  // p1s
     2.343833e-03,  // p2
    -7.910379e-02   // k3
    );

    
    
    // Get all image files
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }
    std::sort(image_files.begin(), image_files.end());
    
    std::cout << "\nFound " << image_files.size() 
              << " images in dataset" << std::endl;
    std::cout << "Processing first 20 frames..." << std::endl;
    std::cout << "Press any key to advance, ESC to quit\n" << std::endl;
    
    // Process frames
    int num_frames = std::min(20, (int)image_files.size());


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
    
    for (int i = 0; i < num_frames; i++) {
        // Load image
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        
        if (image.empty()) {
            std::cerr << "Failed to load: " << image_files[i] << std::endl;
            continue;
        }
        
        // Create frame (timestamp = index * 0.1s for KITTI)
        auto frame = std::make_shared<Frame>(i, i * 0.1, image, camera);

        //Rectification
        frame->rectifyImage();
        
        // Extract ORB features
        frame->extractFeatures(1000);
        
        // Visualize
        cv::Mat img_with_features = frame->drawKeypoints();
        
        // Add frame info
        std::string info = "Frame " + std::to_string(i) + 
                          " | Features: " + std::to_string(frame->keypoints().size());
        cv::putText(img_with_features, info, 
                   cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 
                   1.0, cv::Scalar(0, 255, 0), 2);
        
        // Show
        cv::imshow("KITTI - Feature Extraction", img_with_features);
        
        int key = cv::waitKey(0);  // Wait for key press
        if (key == 27) break;      // ESC to quit
    }
    
    cv::destroyAllWindows();
    
    std::cout << "\n✅ Test completed successfully!" << std::endl;
    std::cout << "Next steps:" << std::endl;
    std::cout << "  1. Feature matching (coming tomorrow)" << std::endl;
    std::cout << "  2. Visual odometry" << std::endl;
    std::cout << "  3. SLAM!" << std::endl;
    
    return 0;
}