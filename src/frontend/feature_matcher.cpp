#include "frontend/feature_matcher.h"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace semantic_vslam {

FeatureMatcher::FeatureMatcher(float ratio_threshold, float max_distance)
    : ratio_threshold_(ratio_threshold), max_distance_(max_distance) {
    
    // BFMatcher for ORB (binary descriptors, use Hamming distance)
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    
    std::cout << "FeatureMatcher created:" << std::endl;
    std::cout << "  Ratio threshold: " << ratio_threshold_ << std::endl;
    std::cout << "  Max distance: " << max_distance_ << std::endl;
}

std::vector<FeatureMatcher::Match> FeatureMatcher::match(
    const Frame::Ptr& frame1,
    const Frame::Ptr& frame2) {
    
    if (frame1->descriptors().empty() || frame2->descriptors().empty()) {
        std::cerr << "ERROR: One or both frames have no descriptors!" << std::endl;
        return {};
    }
    
    // knnMatch: for each descriptor in frame1, find 2 best matches in frame2
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(
        frame1->descriptors(), 
        frame2->descriptors(), 
        knn_matches, 
        2  // k=2 for ratio test
    );
    
    // Apply Lowe's ratio test
    std::vector<Match> good_matches;
    good_matches.reserve(knn_matches.size());
    
    for (size_t i = 0; i < knn_matches.size(); i++) {
        // Need at least 2 matches for ratio test
        if (knn_matches[i].size() < 2) continue;
        
        const cv::DMatch& best = knn_matches[i][0];
        const cv::DMatch& second = knn_matches[i][1];
        
        // Lowe's ratio test: best_dist / second_dist < threshold
        // This filters ambiguous matches
        if (best.distance < ratio_threshold_ * second.distance &&
            best.distance < max_distance_) {
            
            good_matches.emplace_back(
                best.queryIdx,  // index in frame1
                best.trainIdx,  // index in frame2
                best.distance
            );
        }
    }
    
    std::cout << "Frame " << frame1->id() << " -> " << frame2->id() 
              << ": " << knn_matches.size() << " candidates, "
              << good_matches.size() << " good matches" << std::endl;
    
    return good_matches;
}

cv::Mat FeatureMatcher::visualizeMatches(
    const Frame::Ptr& frame1,
    const Frame::Ptr& frame2,
    const std::vector<Match>& matches) const {
    
    // Convert our Match format to OpenCV DMatch format
    std::vector<cv::DMatch> cv_matches;
    cv_matches.reserve(matches.size());
    
    for (const auto& m : matches) {
        cv_matches.emplace_back(m.idx1, m.idx2, m.distance);
    }
    
    // Draw matches
    cv::Mat img_matches;
    cv::drawMatches(
        frame1->image(), frame1->keypoints(),
        frame2->image(), frame2->keypoints(),
        cv_matches, img_matches,
        cv::Scalar(0, 255, 0),      // Green lines
        cv::Scalar(255, 0, 0),      // Blue keypoints
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );
    
    // Add text overlay
    std::string info = "Frame " + std::to_string(frame1->id()) + 
                      " -> " + std::to_string(frame2->id()) +
                      " | Matches: " + std::to_string(matches.size());
    
    cv::putText(img_matches, info,
               cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX,
               0.8, cv::Scalar(0, 255, 0), 2);
    
    return img_matches;
}

void FeatureMatcher::printMatchStats(const std::vector<Match>& matches) const {
    if (matches.empty()) {
        std::cout << "  No matches!" << std::endl;
        return;
    }
    
    // Compute statistics
    std::vector<float> distances;
    distances.reserve(matches.size());
    for (const auto& m : matches) {
        distances.push_back(m.distance);
    }
    
    std::sort(distances.begin(), distances.end());
    
    float min_dist = distances.front();
    float max_dist = distances.back();
    float median_dist = distances[distances.size() / 2];
    float mean_dist = std::accumulate(distances.begin(), distances.end(), 0.0f) 
                     / distances.size();
    
    std::cout << "  Match statistics:" << std::endl;
    std::cout << "    Count: " << matches.size() << std::endl;
    std::cout << "    Distance - Min: " << min_dist 
              << ", Max: " << max_dist
              << ", Mean: " << mean_dist
              << ", Median: " << median_dist << std::endl;
}

} // namespace semantic_vslam