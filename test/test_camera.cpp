#include "core/camera.h"
#include "core/frame.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <string>

namespace fs = std::filesystem;
using namespace semantic_vslam;

static cv::Mat toBGR(const cv::Mat& gray) {
    cv::Mat bgr;
    if (gray.channels() == 1) cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    else bgr = gray.clone();
    return bgr;
}

int main(int argc, char** argv) {
    std::cout << "=== Check if KITTI images are already rectified ===\n";

    // Paths
    std::string data_root = "../data/dataset/sequences/00";
    std::string image_dir = data_root + "/image_0";

    if (!fs::exists(image_dir)) {
        std::cerr << "ERROR: Image directory not found: " << image_dir << "\n";
        return -1;
    }

    // Collect images
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) {
        std::cerr << "ERROR: No .png images found in " << image_dir << "\n";
        return -1;
    }

    std::cout << "Found " << image_files.size() << " images.\n";
    int num_frames = std::min(20, (int)image_files.size());

    // Load first image to get size
    cv::Mat first = cv::imread(image_files[0], cv::IMREAD_GRAYSCALE);
    if (first.empty()) {
        std::cerr << "ERROR: Could not read first image: " << image_files[0] << "\n";
        return -1;
    }
    int width  = first.cols;
    int height = first.rows;

    // --- Choose calibration ---
    // Option A: Your RAW-like calibration (K_00, D_00 you pasted)
   

    // Option B (recommended for KITTI Odometry sequences images that are already rectified):
    // Uncomment this and comment Option A above to test "no change".
    
    cv::Mat D = cv::Mat::zeros(1,5,CV_64F);
    double fx = 718.856;
    double fy = 718.856;
    double cx = 607.1928;
    double cy = 185.2157;
    

    auto camera = std::make_shared<Camera>(fx, fy, cx, cy, D, width, height);
    camera->precomputeUndistortMaps();

    std::cout << "Image size: " << width << " x " << height << "\n";
    std::cout << "Camera fx,fy,cx,cy: " << fx << ", " << fy << ", " << cx << ", " << cy << "\n";
    std::cout << "Has undistort maps: " << (camera->hasUndistortMaps() ? "YES" : "NO") << "\n\n";

    std::cout << "Controls: any key = next frame, ESC = quit\n\n";

    for (int i = 0; i < num_frames; i++) {
        cv::Mat img = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Failed to load: " << image_files[i] << "\n";
            continue;
        }

        cv::Mat original = img.clone();

        auto frame = std::make_shared<Frame>(i, i * 0.1, img, camera);

        // Apply rectification/undistortion (if maps exist, else should do nothing)
        frame->rectifyImage();

        // Requires: Frame::image() getter
        cv::Mat rectified = frame->image().clone();

        // Diff + MAE
        cv::Mat diff;
        cv::absdiff(original, rectified, diff);
        double mae = cv::mean(diff)[0];

        // Visualize original | rectified | diff
        cv::Mat original_bgr = toBGR(original);
        cv::Mat rectified_bgr = toBGR(rectified);

        // Diff visualization: amplify for visibility
        cv::Mat diff_vis;
        diff.convertTo(diff_vis, CV_8U);              // already 8U but safe
        cv::Mat diff_bgr = toBGR(diff_vis);

        cv::putText(original_bgr, "Original", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
        cv::putText(rectified_bgr, "Rectified", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);

        std::string diff_text = "Diff | MAE=" + std::to_string(mae);
        cv::putText(diff_bgr, diff_text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);

        cv::Mat left, vis;
        cv::hconcat(original_bgr, rectified_bgr, left);
        cv::hconcat(left, diff_bgr, vis);

        std::cout << "Frame " << i << " MAE(absdiff) = " << mae << "\n";

        cv::imshow("Original | Rectified | Diff", vis);
        int key = cv::waitKey(0);
        if (key == 27) break; // ESC
    }

    cv::destroyAllWindows();
    std::cout << "\n Done.\n";
    return 0;
}
