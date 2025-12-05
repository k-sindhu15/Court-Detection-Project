#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"  ✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("  ✓ Ultralytics YOLO")
    except ImportError as e:
        print(f"  ✗ Ultralytics: {e}")
        return False
    
    try:
        from src.court_detection import CourtLineDetector, CourtRegionDetector, CourtMasker
        print("  ✓ Court detection modules")
    except ImportError as e:
        print(f"  ✗ Court detection: {e}")
        return False
    
    try:
        from src.player_tracking import PlayerDetector, PlayerTracker
        print("  ✓ Player tracking modules")
    except ImportError as e:
        print(f"  ✗ Player tracking: {e}")
        return False
    
    try:
        from src.video_processing import VideoReader, VideoWriter
        print("  ✓ Video processing modules")
    except ImportError as e:
        print(f"  ✗ Video processing: {e}")
        return False
    
    try:
        from src.pipeline import BadmintonCourtProcessor
        print("  ✓ Main pipeline")
    except ImportError as e:
        print(f"  ✗ Pipeline: {e}")
        return False
    
    return True


def test_yolo_model():
    """Test YOLO model loading."""
    print("\nTesting YOLO model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("  ✓ YOLOv8n model loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ YOLO model loading failed: {e}")
        return False


def test_basic_detection():
    """Test basic detection on a synthetic image."""
    print("\nTesting basic detection...")
    
    try:
        import numpy as np
        import cv2
        from src.court_detection import CourtLineDetector
        
        # Create a synthetic court image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)  # Green background
        
        # Draw court lines
        cv2.rectangle(img, (100, 100), (540, 380), (255, 255, 255), 2)
        cv2.line(img, (100, 240), (540, 240), (255, 255, 255), 2)
        cv2.line(img, (320, 100), (320, 380), (255, 255, 255), 2)
        
        # Test line detection
        detector = CourtLineDetector()
        detector.calibrate([img])
        lines, mask = detector.detect_lines(img)
        
        print(f"  ✓ Detected {len(lines)} lines")
        return True
    except Exception as e:
        print(f"  ✗ Basic detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.utils import ConfigManager
        
        config = ConfigManager()
        method = config.get('court_detection.method')
        print(f"  ✓ Configuration loaded, court detection method: {method}")
        return True
    except Exception as e:
        print(f"  ✗ Configuration failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Badminton Court Detection - Installation Test")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Basic Detection", test_basic_detection()))
    results.append(("YOLO Model", test_yolo_model()))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✓ All tests passed! The system is ready to use.")
        print("\nUsage:")
        print("  python main.py input_video.mp4 output_video.mp4")
    else:
        print("\n✗ Some tests failed. Please check the installation.")
        print("\nTry running:")
        print("  pip install -r requirements.txt")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
