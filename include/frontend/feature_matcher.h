#pragma once

#include "core/frame.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace semantic_vslam {

class FeatureMatcher {
public:
    struct Match {
        int idx1;       // Keypoint index in frame1
        int idx2;       // Keypoint index in frame2
        float distance; // Descriptor distance (Hamming for ORB)
        
        Match(int i1, int i2, float d) 
            : idx1(i1), idx2(i2), distance(d) {}
    };
    
    // Constructor

    //ratio = que tan mejor es la primera opcion en comparacion a la segunda
    // max _distance = que tanto se parece
    FeatureMatcher(float ratio_threshold = 0.75, 
                   float max_distance = 50.0);
    
    // Match features between two frames using Lowe's ratio test
    std::vector<Match> match(
        const Frame::Ptr& frame1,
        const Frame::Ptr& frame2
    );
    
    // Visualize matches between two frames
    cv::Mat visualizeMatches(
        const Frame::Ptr& frame1,
        const Frame::Ptr& frame2,
        const std::vector<Match>& matches
    ) const;
    
    // Get match statistics
    void printMatchStats(const std::vector<Match>& matches) const;

private:
    cv::Ptr<cv::BFMatcher> matcher_;
    float ratio_threshold_;   // Lowe's ratio test threshold (0.7-0.8)
    float max_distance_;      // Max descriptor distance
};

} // namespace semantic_vslam