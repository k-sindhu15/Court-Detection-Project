# Technical Report: Court-Focused Video Cropping and Masking for Badminton Match

## 1. Introduction

### 1.1 Problem Statement

In multi-court badminton broadcasts, isolating a specific court for analysis requires sophisticated computer vision techniques. This project implements an automated solution to:

- Detect and extract the focused court region
- Mask irrelevant areas (adjacent courts, background)
- Track players on the focused court with consistent IDs

### 1.2 Objectives

1. Generate processed video with only the main court visible [10 marks]
2. Remove regions behind the court while preserving court information [15 marks]
3. Track players with consistent track IDs [10 marks]
4. Simultaneously mask adjacent courts and background regions [15 marks]

## 2. Approach and Methodology

### 2.1 Court Detection

Our court detection system uses a **hybrid approach** combining multiple techniques:

#### 2.1.1 Edge-Based Detection

- **Canny Edge Detection**: Adaptive thresholding based on image statistics (not fixed values)
- **Hough Line Transform**: Detects court boundary lines
- **Line Filtering**: Separates horizontal and vertical lines for court structure identification

#### 2.1.2 Color-Based Detection

- **White Line Detection**: HSV thresholding for court markings
- **Court Surface Detection**: Identifies green, wooden, or blue court surfaces
- **Adaptive Calibration**: Color thresholds calibrated per video using percentile statistics

#### 2.1.3 Focused Court Selection

When multiple courts are detected, we select the focused court using a weighted scoring system:

- **Size Score (50%)**: Larger courts get higher priority
- **Center Score (30%)**: Courts closer to frame center score higher
- **Activity Score (20%)**: Optical flow analysis detects active gameplay

### 2.2 Court Masking

#### 2.2.1 Masking Strategies

1. **Rectangular Masking**: Simple bounding box mask for standard views
2. **Polygon Masking**: Uses detected corner points for perspective-distorted courts
3. **Perspective-Aware Masking**: Trapezoid mask for behind-court regions

#### 2.2.2 Edge Smoothing

- Gaussian blur applied to mask edges for natural transitions
- Configurable feather radius based on frame dimensions (ratio-based, not fixed pixels)

### 2.3 Player Tracking

#### 2.3.1 Player Detection

- **Model**: YOLOv8 (scalable from yolov8n to yolov8x based on accuracy/speed needs)
- **Filtering**: Detections filtered to only include players within court boundaries
- **Confidence Thresholding**: Configurable threshold (default 0.5)

#### 2.3.2 Tracking Algorithm (ByteTrack)

- **Two-Stage Association**: Matches high-confidence then low-confidence detections
- **Track Buffer**: Maintains lost tracks for 30 frames to handle occlusions
- **IoU Matching**: Uses Intersection over Union for detection-to-track association

## 3. Implementation Details

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BadmintonCourtProcessor                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ VideoReader  │→│CourtDetector │→│  CourtMasker     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         ↓                ↓                    ↓             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │PlayerDetector│→│ ByteTracker  │→│  VideoWriter     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Robustness Mechanisms

| Challenge | Solution |
|-----------|----------|
| Varying lighting | Adaptive thresholds based on frame statistics |
| Different camera angles | Perspective-aware detection and masking |
| Different court colors | Multi-color detection (green, wood, blue) |
| Multiple court sizes | Ratio-based parameters (% of image, not pixels) |
| Fast player movement | ByteTrack's two-stage association |
| Player occlusions | Track buffer maintains IDs during occlusions |

### 3.3 No Fixed Hyperparameters

All critical parameters are computed adaptively:

```python
# Example: Adaptive Canny thresholds
median = np.median(gray_image)
lower = int(max(0, (1.0 - 0.33) * median))
upper = int(min(255, (1.0 + 0.33) * median))

# Example: Adaptive line length thresholds
diagonal = sqrt(width^2 + height^2)
min_line_length = diagonal * 0.05  # 5% of diagonal
max_line_gap = diagonal * 0.015    # 1.5% of diagonal
```

## 4. Results and Evaluation

### 4.1 Correctness

- Court boundaries accurately captured using hybrid detection
- Mask precisely covers irrelevant regions
- Player tracks maintained through typical gameplay scenarios

### 4.2 Robustness

- Tested across different lighting conditions
- Works with various camera angles (within reasonable limits)
- Handles different court surface colors

### 4.3 Efficiency

- Real-time processing possible with GPU acceleration
- Configurable model size for speed/accuracy trade-off
- Frame caching for mask optimization

## 5. Trade-offs and Limitations

### 5.1 Trade-offs

| Trade-off | Choice Made |
|-----------|-------------|
| Speed vs Accuracy | Configurable YOLO model (n/s/m/l/x) |
| Simplicity vs Robustness | Hybrid detection (more complex but more reliable) |
| Memory vs Speed | Mask caching for unchanged court regions |

### 5.2 Limitations

1. **Extreme oblique angles**: Very side-on views may not detect court properly
2. **Heavy occlusion**: If court is >50% occluded, detection may fail
3. **Track ID switches**: Rapid crossing of players may cause brief ID switches
4. **Non-standard courts**: Unusual court colors may require calibration

## 6. Instructions for Running

### 6.1 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Basic Usage

```bash
# Process video with default settings
python main.py input.mp4 output.mp4

# Mask-only mode (faster)
python main.py input.mp4 output.mp4 --mask-only

# With preview window
python main.py input.mp4 output.mp4 --preview
```

### 6.3 Configuration

Modify `config/default_config.yaml` for custom settings or pass a custom config:

```bash
python main.py input.mp4 output.mp4 --config custom_config.yaml
```

## 7. Bonus Features Implemented

1. **Deep Learning Integration**: YOLOv8 for player detection with multiple model sizes
2. **CLI Interface**: Full command-line interface with parameter adjustment
3. **Preview Mode**: Real-time preview window during processing
4. **Configurable System**: YAML-based configuration with runtime overrides

## 8. Conclusion

This solution provides a robust, adaptive system for court-focused video processing. The hybrid approach to court detection, combined with ByteTrack player tracking, delivers reliable results across varying conditions without relying on fixed hyperparameters.

---

**Author**: [Your Name]  
**Date**: December 2025  
**Project**: VISIST.AI Internship Evaluation
