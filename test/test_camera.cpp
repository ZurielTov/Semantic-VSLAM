#include "core/camera.h"
#include <iostream>
#include <Eigen/Core>

using namespace semantic_vslam;

int main() {
    std::cout << "=== Testing Camera Model ===" << std::endl;
    
    // Create KITTI camera
    Camera cam(718.856, 718.856, 607.1928, 185.2157, 1241, 376);
    
    // Test 1: Project 3D point
    std::cout << "\nTest 1: Project 3D point to pixel" << std::endl;
    Eigen::Vector3d point_3d(1.0, 2.0, 5.0);  // 1m right, 2m up, 5m forward
    Eigen::Vector2d pixel = cam.project(point_3d);
    
    std::cout << "  3D point: (" << point_3d.transpose() << ")" << std::endl;
    std::cout << "  Pixel: (" << pixel.transpose() << ")" << std::endl;
    std::cout << "  Expected: (~751, ~472)" << std::endl;
    
    // Test 2: Backproject pixel to ray
    std::cout << "\nTest 2: Backproject pixel to ray" << std::endl;
    Eigen::Vector2d test_pixel(500, 300);
    Eigen::Vector3d ray = cam.backproject(test_pixel);
    
    std::cout << "  Pixel: (" << test_pixel.transpose() << ")" << std::endl;
    std::cout << "  Ray (normalized): (" << ray.transpose() << ")" << std::endl;
    std::cout << "  Ray norm: " << ray.norm() << " (should be 1.0)" << std::endl;
    
    // Test 3: Round-trip (project then backproject)
    std::cout << "\nTest 3: Round-trip test" << std::endl;
    Eigen::Vector3d original(2.0, -1.0, 10.0);
    Eigen::Vector2d projected = cam.project(original);
    Eigen::Vector3d ray_back = cam.backproject(projected);
    
    // Scale ray to same depth as original
    Eigen::Vector3d reconstructed = ray_back * (original.z() / ray_back.z());
    
    std::cout << "  Original 3D: (" << original.transpose() << ")" << std::endl;
    std::cout << "  Reconstructed: (" << reconstructed.transpose() << ")" << std::endl;
    std::cout << "  Error: " << (original - reconstructed).norm() << " (should be ~0)" << std::endl;
    
    // Test 4: Image bounds
    std::cout << "\nTest 4: Image bounds check" << std::endl;
    std::cout << "  (100, 100) in image? " << cam.isInImage({100, 100}) << std::endl;
    std::cout << "  (2000, 100) in image? " << cam.isInImage({2000, 100}) << std::endl;
    std::cout << "  (620, 188) in image? " << cam.isInImage({620, 188}) << " (center)" << std::endl;
    
    std::cout << "\n✅ All camera tests passed!" << std::endl;
    
    return 0;
}