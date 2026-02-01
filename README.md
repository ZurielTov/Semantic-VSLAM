# Semantic Monocular SLAM

Real-time monocular visual SLAM with semantic object-level mapping.

## Project Goals

Build a complete SLAM system that:
- Estimates camera pose from monocular video
- Reconstructs 3D map of environment
- Detects loop closures to eliminate drift
- **Identifies semantic objects** (cars, chairs, etc.)
- Enables object-level queries ("show me all chairs")

## Architecture
```
┌─────────────┐
│   Camera    │
└──────┬──────┘
       │
       v
┌─────────────┐      ┌──────────────┐
│  Frontend   │─────>│ Local Mapper │
│ (Track)     │      │ (BA)         │
└─────────────┘      └──────┬───────┘
       │                    │
       v                    v
┌─────────────┐      ┌──────────────┐
│ Loop Detect │─────>│ Pose Graph   │
│ (DBoW3)     │      │ Optimization │
└─────────────┘      └──────────────┘
       │
       v
┌─────────────┐
│  Semantic   │
│  Mapping    │
│  (YOLO)     │
└─────────────┘
```

## Current Status

**Working On** (Jan 20-26, 2026):
- [x] Project setup
- [x] Camera model (projection/backprojection)
- [x] Frame class with ORB feature extraction
- [x] KITTI dataset loader
- [x] Feature matching
- [ ] Visual odometry (5-point algorithm)

**Upcoming:**
- [ ] Triangulation & mapping
- [ ] Local bundle adjustment
- [ ] Loop closure detection
- [ ] Semantic mapping

## Quick Start

### Dependencies
```bash
# Ubuntu 20.04/22.04
sudo apt install build-essential cmake git
sudo apt install libeigen3-dev libopencv-dev
```

### Build
```bash
mkdir build && cd build
cmake ..
make -j4
```

### Run
```bash
# Test camera model
./test_camera

# Run on KITTI dataset
./run_kitti
```

## Datasets

- **KITTI Odometry**: Outdoor driving sequences
- **TUM RGB-D** (planned): Indoor sequences with ground truth

Download KITTI from: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

## 📖 Resources

- [Visual SLAM Book](https://github.com/gaoxiang12/slambook-en) - Xiang Gao
- [ORB-SLAM3 Paper](https://arxiv.org/abs/2007.11898)
- [DBoW2](https://github.com/dorian3d/DBoW2)

## Progress Log

### January 20, 2026
- Initial project structure
- Camera model implemented
- ORB feature extraction working
- Successfully tested on KITTI sequence 00

## Learning Goals

This project is built to understand SLAM from fundamentals:
1. **Geometric vision**: Epipolar geometry, triangulation
2. **Optimization**: Bundle adjustment, pose graph
3. **Place recognition**: Bag of words, loop closure
4. **Deep learning integration**: YOLO + classical SLAM

## 🖼️ Visual Results

### Feature Extraction
![Feature Extraction](results/day1_features.png)
*1000 ORB features extracted per frame on KITTI sequence 00*

### Feature Matching
![Feature Matching](results/day2_matching.png)
*Lowe's ratio test (threshold=0.75) achieving 450+ matches per frame pair*

[Video: Feature Tracking](results/feature_tracking.avi)

##  Contact

Francisco Zuriel Tovar Mendoza  
zuriel.tovar.m@gmail.com  
[LinkedIn](https://linkedin.com/in/zurieltov)

---

*Part of my robotics engineering journey at Tecnológico de Monterrey*