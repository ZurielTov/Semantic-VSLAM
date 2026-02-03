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
    std::cout << "=== Generating Feature Tracking Video ===" << std::endl;
    
    // Paths
    std::string data_root = "../data/dataset/sequences/00";
    std::string image_dir = data_root + "/image_0";
    std::string output_video = "../results/feature_tracking.avi";

    //Create Dist coeff
    cv::Mat D = (cv::Mat_<double>(1,5) <<
    -3.791375e-01,  // k1
     2.148119e-01,  // k2
     1.227094e-03,  // p1s
     2.343833e-03,  // p2
    -7.910379e-02   // k3
    );

    
    // Get images
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) {
    std::cerr << "ERROR: No images found in " << image_dir << std::endl;
    return -1;
    }

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


    // Create matcher
    FeatureMatcher matcher(0.75, 50.0);



    // Video writer
    cv::VideoWriter video;
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = 10.0;  // 10 fps for visualization
    cv::Size frame_size(width * 2, height);

    
    video.open(output_video, codec, fps, frame_size, true);
    
    if (!video.isOpened()) {
        std::cerr << "ERROR: Could not open video writer!" << std::endl;
        return -1;
    }
    
    std::cout << "Processing " << std::min(100, (int)image_files.size()) 
              << " frame pairs..." << std::endl;
    
    Frame::Ptr prev_frame = nullptr;
    int match_count_sum = 0;
    int pair_count = 0;
    
    for (int i = 0; i < std::min(100, (int)image_files.size()); i++) {
        // Load image
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        
        // Create frame
        auto curr_frame = std::make_shared<Frame>(i, i * 0.1, image, camera);

        //Rectification
        curr_frame->rectifyImage();

        curr_frame->extractFeatures(1000);
        
        // Match
        if (prev_frame) {
            auto matches = matcher.match(prev_frame, curr_frame);
            
            // Stats
            match_count_sum += matches.size();
            pair_count++;
            
            // Visualize
            cv::Mat img_matches = matcher.visualizeMatches(
                prev_frame, curr_frame, matches
            );
            
            // Write to video
            video.write(img_matches);
            
            // Progress
            if (i % 10 == 0) {
                std::cout << "  Processed " << i << " frames..." << std::endl;
            }
        }
        
        prev_frame = curr_frame;
    }
    
    video.release();
    
    // Statistics
    float avg_matches = (float)match_count_sum / pair_count;
    
    std::cout << "\n✅ Video generation complete!" << std::endl;
    std::cout << "Output: " << output_video << std::endl;
    std::cout << "Statistics:" << std::endl;
    std::cout << "  Total frame pairs: " << pair_count << std::endl;
    std::cout << "  Average matches per pair: " << avg_matches << std::endl;
    std::cout << "  Video FPS: " << fps << std::endl;
    
    return 0;
}