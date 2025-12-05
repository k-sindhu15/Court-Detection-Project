"""
Court Line Detector

Detects badminton court lines using multiple techniques:
- Edge detection (Canny + Hough Transform)
- Color segmentation (for white/yellow court lines)
- Adaptive thresholding

This module is designed to be robust across different lighting conditions
and camera angles without relying on fixed hyperparameters.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Line:
    """Represents a detected line segment."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def length(self) -> float:
        """Calculate line length."""
        return np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
    
    @property
    def angle(self) -> float:
        """Calculate line angle in degrees."""
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """Get line midpoint."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    def is_horizontal(self, tolerance: float = 15) -> bool:
        """Check if line is approximately horizontal."""
        angle = abs(self.angle)
        return angle < tolerance or angle > (180 - tolerance)
    
    def is_vertical(self, tolerance: float = 15) -> bool:
        """Check if line is approximately vertical."""
        angle = abs(self.angle)
        return 90 - tolerance < angle < 90 + tolerance


class CourtLineDetector:
    """
    Detects court lines using adaptive computer vision techniques.
    
    The detector automatically calibrates its parameters based on
    image statistics, making it robust across different videos.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the court line detector.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self._calibrated = False
        self._calibration_data = {}
    
    def calibrate(self, frames: List[np.ndarray]) -> None:
        """
        Calibrate detector parameters from sample frames.
        
        Args:
            frames: List of sample frames for calibration
        """
        if not frames:
            logger.warning("No frames provided for calibration")
            return
        
        # Analyze image statistics
        brightness_values = []
        contrast_values = []
        white_pixel_ratios = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness (mean intensity)
            brightness_values.append(np.mean(gray))
            
            # Contrast (standard deviation)
            contrast_values.append(np.std(gray))
            
            # White pixel ratio
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_ratio = np.sum(binary > 0) / binary.size
            white_pixel_ratios.append(white_ratio)
        
        self._calibration_data = {
            'mean_brightness': np.mean(brightness_values),
            'mean_contrast': np.mean(contrast_values),
            'white_pixel_ratio': np.mean(white_pixel_ratios),
            'frame_height': frames[0].shape[0],
            'frame_width': frames[0].shape[1],
            'diagonal': np.sqrt(frames[0].shape[0]**2 + frames[0].shape[1]**2)
        }
        
        self._calibrated = True
        logger.info(f"Calibrated with data: {self._calibration_data}")
    
    def _get_adaptive_canny_thresholds(self, gray: np.ndarray) -> Tuple[int, int]:
        """
        Calculate adaptive Canny edge detection thresholds.
        
        Uses Otsu's method and image statistics for robust thresholding.
        """
        # Use Otsu's threshold as a baseline
        otsu_thresh, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Calculate percentile-based thresholds
        median = np.median(gray)
        sigma = 0.33
        
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        # Blend with Otsu threshold
        lower = int(0.5 * lower + 0.5 * otsu_thresh * 0.5)
        upper = int(0.5 * upper + 0.5 * otsu_thresh)
        
        return lower, upper
    
    def _get_adaptive_hough_params(self) -> Dict[str, int]:
        """Get adaptive Hough transform parameters based on calibration."""
        if not self._calibrated:
            # Default parameters for 1080p
            return {
                'threshold': 100,
                'min_line_length': 100,
                'max_line_gap': 20
            }
        
        diagonal = self._calibration_data['diagonal']
        
        return {
            'threshold': int(diagonal * 0.05),
            'min_line_length': int(diagonal * 0.05),
            'max_line_gap': int(diagonal * 0.015)
        }
    
    def detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect edges using adaptive Canny edge detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary edge image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Get adaptive thresholds
        lower, upper = self._get_adaptive_canny_thresholds(blurred)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, lower, upper)
        
        # Dilate edges slightly to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def detect_lines_hough(self, edges: np.ndarray) -> List[Line]:
        """
        Detect lines using Hough Line Transform.
        
        Args:
            edges: Binary edge image
            
        Returns:
            List of detected lines
        """
        params = self._get_adaptive_hough_params()
        
        lines_array = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=params['threshold'],
            minLineLength=params['min_line_length'],
            maxLineGap=params['max_line_gap']
        )
        
        if lines_array is None:
            return []
        
        lines = []
        for line in lines_array:
            x1, y1, x2, y2 = line[0]
            lines.append(Line(x1, y1, x2, y2))
        
        return lines
    
    def detect_white_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect white court lines using color segmentation.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask of white lines
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adaptive white detection based on frame statistics
        v_channel = hsv[:, :, 2]
        v_thresh = np.percentile(v_channel, 90)  # Top 10% brightest
        
        # White line detection (high value, low saturation)
        lower_white = np.array([0, 0, int(v_thresh)])
        upper_white = np.array([180, 50, 255])
        
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        return white_mask
    
    def detect_court_surface(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect court surface (typically green or wooden color).
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask of court surface
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green court detection
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Wooden/brown court detection
        lower_wood = np.array([10, 30, 80])
        upper_wood = np.array([30, 150, 200])
        wood_mask = cv2.inRange(hsv, lower_wood, upper_wood)
        
        # Blue court detection (some indoor courts)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine all court surface masks
        court_mask = cv2.bitwise_or(green_mask, wood_mask)
        court_mask = cv2.bitwise_or(court_mask, blue_mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel)
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN, kernel)
        
        return court_mask
    
    def detect_lines(self, frame: np.ndarray) -> Tuple[List[Line], np.ndarray]:
        """
        Main method to detect all court lines.
        
        Combines edge detection and color segmentation for robust detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (detected lines, combined line mask)
        """
        # Edge-based detection
        edges = self.detect_edges(frame)
        edge_lines = self.detect_lines_hough(edges)
        
        # Color-based detection
        white_mask = self.detect_white_lines(frame)
        
        # Combine edge and color information
        combined_mask = cv2.bitwise_or(edges, white_mask)
        
        # Detect lines in combined mask
        combined_lines = self.detect_lines_hough(combined_mask)
        
        # Merge and filter lines
        all_lines = self._merge_similar_lines(edge_lines + combined_lines)
        
        return all_lines, combined_mask
    
    def _merge_similar_lines(
        self, 
        lines: List[Line], 
        distance_threshold: float = 20,
        angle_threshold: float = 10
    ) -> List[Line]:
        """
        Merge similar/duplicate lines.
        
        Args:
            lines: List of detected lines
            distance_threshold: Max distance between line midpoints to merge
            angle_threshold: Max angle difference to merge
            
        Returns:
            Merged list of lines
        """
        if not lines:
            return []
        
        merged = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            
            # Collect similar lines
            similar = [line1]
            used[i] = True
            
            for j, line2 in enumerate(lines[i + 1:], start=i + 1):
                if used[j]:
                    continue
                
                # Check angle similarity
                angle_diff = abs(line1.angle - line2.angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff > angle_threshold:
                    continue
                
                # Check midpoint distance
                mid1 = line1.midpoint
                mid2 = line2.midpoint
                dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                
                if dist < distance_threshold:
                    similar.append(line2)
                    used[j] = True
            
            # Average similar lines
            if similar:
                avg_x1 = int(np.mean([l.x1 for l in similar]))
                avg_y1 = int(np.mean([l.y1 for l in similar]))
                avg_x2 = int(np.mean([l.x2 for l in similar]))
                avg_y2 = int(np.mean([l.y2 for l in similar]))
                merged.append(Line(avg_x1, avg_y1, avg_x2, avg_y2))
        
        return merged
    
    def filter_court_lines(
        self, 
        lines: List[Line],
        frame_shape: Tuple[int, int]
    ) -> Tuple[List[Line], List[Line]]:
        """
        Separate lines into horizontal and vertical court lines.
        
        Args:
            lines: All detected lines
            frame_shape: (height, width) of frame
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines)
        """
        height, width = frame_shape[:2]
        min_length = min(height, width) * 0.05  # Minimum line length
        
        horizontal = []
        vertical = []
        
        for line in lines:
            if line.length < min_length:
                continue
            
            if line.is_horizontal():
                horizontal.append(line)
            elif line.is_vertical():
                vertical.append(line)
        
        # Sort by position
        horizontal.sort(key=lambda l: l.midpoint[1])  # Sort by y
        vertical.sort(key=lambda l: l.midpoint[0])    # Sort by x
        
        return horizontal, vertical
    
    def visualize_lines(
        self, 
        frame: np.ndarray, 
        lines: List[Line],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detected lines on frame.
        
        Args:
            frame: Input frame
            lines: Lines to draw
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with lines drawn
        """
        result = frame.copy()
        
        for line in lines:
            cv2.line(
                result,
                (line.x1, line.y1),
                (line.x2, line.y2),
                color,
                thickness
            )
        
        return result
