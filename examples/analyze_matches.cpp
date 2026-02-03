#include "core/camera.h"
#include "core/frame.h"
#include "frontend/feature_matcher.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>

namespace fs = std::filesystem;
using namespace semantic_vslam;

int main() {
    std::cout << "=== Match Quality Analysis ===" << std::endl;
    
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
    
    // Test different ratio thresholds
    std::vector<float> ratio_thresholds = {0.6, 0.7, 0.75, 0.8, 0.85, 0.9};
    
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
    
    std::cout << "\nTesting ratio thresholds on first 50 frame pairs\n" << std::endl;
    
    // Results file
    std::ofstream results_file("../results/match_analysis.txt");
    results_file << "Ratio_Threshold,Avg_Matches,Min_Matches,Max_Matches" << std::endl;
    
    for (float ratio : ratio_thresholds) {
        FeatureMatcher matcher(ratio, 50.0);
        
        std::vector<int> match_counts;
        Frame::Ptr prev_frame = nullptr;
        
        for (int i = 0; i < std::min(50, (int)image_files.size()); i++) {
            cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
            if (image.empty()) continue;
            
            auto curr_frame = std::make_shared<Frame>(i, i * 0.1, image, camera);

            //Rectification
            curr_frame->rectifyImage();
            curr_frame->extractFeatures(1000);
            
            if (prev_frame) {
                auto matches = matcher.match(prev_frame, curr_frame);
                match_counts.push_back(matches.size());
            }
            
            prev_frame = curr_frame;
        }
        
        // Statistics
        if (!match_counts.empty()) {
            int sum = std::accumulate(match_counts.begin(), match_counts.end(), 0);
            float avg = (float)sum / match_counts.size();
            int min = *std::min_element(match_counts.begin(), match_counts.end());
            int max = *std::max_element(match_counts.begin(), match_counts.end());
            
            std::cout << "Ratio " << ratio << ": "
                     << "Avg=" << avg << ", "
                     << "Min=" << min << ", "
                     << "Max=" << max << std::endl;
            
            results_file << ratio << "," << avg << "," << min << "," << max << std::endl;
        }
    }
    
    results_file.close();
    
    std::cout << "\n✅ Analysis complete!" << std::endl;
    std::cout << "Results saved to: results/match_analysis.txt" << std::endl;
    std::cout << "\nRecommendation: Use ratio=0.75 for good balance" << std::endl;
    
    return 0;
}