"""
Court Region Detector

Identifies and extracts individual court regions from the video frame.
Uses detected lines and geometric analysis to find court boundaries.
Selects the "focused" court based on various criteria.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging
from .line_detector import CourtLineDetector, Line

logger = logging.getLogger(__name__)


@dataclass
class CourtRegion:
    """Represents a detected court region."""
    
    # Bounding box (x, y, width, height)
    bbox: Tuple[int, int, int, int]
    
    # Corner points (for perspective transform)
    corners: Optional[np.ndarray] = None
    
    # Confidence score
    confidence: float = 0.0
    
    # Region properties
    area: float = 0.0
    center: Tuple[float, float] = (0.0, 0.0)
    
    # Is this the focused/main court?
    is_focused: bool = False
    
    # Activity score (based on motion/players)
    activity_score: float = 0.0
    
    # Mask for this region
    mask: Optional[np.ndarray] = None
    
    @property
    def x(self) -> int:
        return self.bbox[0]
    
    @property
    def y(self) -> int:
        return self.bbox[1]
    
    @property
    def width(self) -> int:
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        return self.bbox[3]
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside the court region."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)


class CourtRegionDetector:
    """
    Detects and identifies court regions in video frames.
    
    This class analyzes detected lines to find rectangular court
    regions and determines which court is the "focus" of the video.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the court region detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.line_detector = CourtLineDetector(config)
        
        # Previous frame for optical flow
        self._prev_gray: Optional[np.ndarray] = None
        
        # Tracking of focused court across frames
        self._focused_court_history: List[Tuple[int, int, int, int]] = []
        self._max_history = 30
    
    def calibrate(self, frames: List[np.ndarray]) -> None:
        """Calibrate the detector using sample frames."""
        self.line_detector.calibrate(frames)
        logger.info("Court region detector calibrated")
    
    def detect_court_contours(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect potential court regions using contour analysis.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of contours representing potential courts
        """
        # Detect court surface
        court_mask = self.line_detector.detect_court_surface(frame)
        
        # Find contours
        contours, _ = cv2.findContours(
            court_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by area (courts should be reasonably large)
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.05  # At least 5% of frame
        
        valid_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > min_area
        ]
        
        return valid_contours
    
    def detect_court_from_lines(
        self, 
        lines: List[Line],
        frame_shape: Tuple[int, int, int]
    ) -> List[CourtRegion]:
        """
        Detect court regions from detected lines.
        
        Uses horizontal and vertical lines to form rectangular regions.
        
        Args:
            lines: Detected court lines
            frame_shape: Shape of the frame
            
        Returns:
            List of detected court regions
        """
        height, width = frame_shape[:2]
        
        # Separate horizontal and vertical lines
        horizontal, vertical = self.line_detector.filter_court_lines(
            lines, (height, width)
        )
        
        courts = []
        
        # Find intersections to form rectangles
        if len(horizontal) >= 2 and len(vertical) >= 2:
            # Find court boundaries from line positions
            h_positions = [l.midpoint[1] for l in horizontal]
            v_positions = [l.midpoint[0] for l in vertical]
            
            # Group lines to find distinct courts
            h_groups = self._cluster_positions(h_positions, threshold=height * 0.1)
            v_groups = self._cluster_positions(v_positions, threshold=width * 0.1)
            
            # Create court regions from line groups
            for h_pair in self._get_pairs(h_groups):
                for v_pair in self._get_pairs(v_groups):
                    y1, y2 = sorted(h_pair)
                    x1, x2 = sorted(v_pair)
                    
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    area = bbox[2] * bbox[3]
                    center = (x1 + bbox[2] / 2, y1 + bbox[3] / 2)
                    
                    # Validate court aspect ratio (approximately 2:1 for badminton)
                    aspect = bbox[2] / bbox[3] if bbox[3] > 0 else 0
                    if 0.3 < aspect < 3.0:  # Allow some flexibility
                        court = CourtRegion(
                            bbox=bbox,
                            area=area,
                            center=center,
                            confidence=0.7
                        )
                        courts.append(court)
        
        return courts
    
    def detect_court_from_contours(
        self, 
        contours: List[np.ndarray],
        frame_shape: Tuple[int, int, int]
    ) -> List[CourtRegion]:
        """
        Create court regions from detected contours.
        
        Args:
            contours: Court surface contours
            frame_shape: Shape of the frame
            
        Returns:
            List of court regions
        """
        courts = []
        
        for cnt in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Get rotated rectangle for better fit
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            
            # Calculate properties
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w / 2, y + h / 2
            
            court = CourtRegion(
                bbox=(x, y, w, h),
                corners=box,
                area=area,
                center=(cx, cy),
                confidence=0.8
            )
            courts.append(court)
        
        return courts
    
    def _cluster_positions(
        self, 
        positions: List[float], 
        threshold: float
    ) -> List[float]:
        """Cluster nearby positions and return cluster centers."""
        if not positions:
            return []
        
        positions = sorted(positions)
        clusters = [[positions[0]]]
        
        for pos in positions[1:]:
            if pos - clusters[-1][-1] < threshold:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        return [np.mean(c) for c in clusters]
    
    def _get_pairs(self, positions: List[float]) -> List[Tuple[float, float]]:
        """Generate all pairs from a list of positions."""
        pairs = []
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1:]:
                pairs.append((p1, p2))
        return pairs
    
    def calculate_activity_score(
        self, 
        frame: np.ndarray, 
        region: CourtRegion
    ) -> float:
        """
        Calculate activity score for a region using optical flow.
        
        Higher activity suggests this is the court being played on.
        
        Args:
            frame: Current frame
            region: Court region to analyze
            
        Returns:
            Activity score (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.5  # Default score
        
        # Extract region of interest
        x, y, w, h = region.bbox
        roi_curr = gray[y:y+h, x:x+w]
        roi_prev = self._prev_gray[y:y+h, x:x+w]
        
        if roi_curr.size == 0 or roi_prev.size == 0:
            return 0.0
        
        # Calculate dense optical flow
        try:
            flow = cv2.calcOpticalFlowFarneback(
                roi_prev, roi_curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate magnitude of flow
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            activity = np.mean(mag)
            
            # Normalize to 0-1 range
            activity_score = min(1.0, activity / 10.0)
            
        except Exception as e:
            logger.debug(f"Optical flow calculation failed: {e}")
            activity_score = 0.5
        
        self._prev_gray = gray
        return activity_score
    
    def select_focused_court(
        self, 
        courts: List[CourtRegion],
        frame_shape: Tuple[int, int, int],
        frame: Optional[np.ndarray] = None
    ) -> Optional[CourtRegion]:
        """
        Select the focused/main court from detected courts.
        
        Selection criteria:
        1. Size (larger courts get higher priority)
        2. Position (courts closer to center get higher priority)
        3. Activity (courts with more motion get higher priority)
        
        Args:
            courts: List of detected court regions
            frame_shape: Shape of the frame
            frame: Current frame (optional, for activity calculation)
            
        Returns:
            The focused court region or None
        """
        if not courts:
            return None
        
        if len(courts) == 1:
            courts[0].is_focused = True
            return courts[0]
        
        height, width = frame_shape[:2]
        frame_center = (width / 2, height / 2)
        frame_area = width * height
        
        # Configuration weights
        size_weight = self.config.get('focus_selection', {}).get('size_weight', 0.5)
        center_weight = self.config.get('focus_selection', {}).get('center_weight', 0.3)
        activity_weight = 0.2
        
        best_court = None
        best_score = -1
        
        for court in courts:
            # Size score (normalized by frame area)
            size_score = court.area / frame_area
            
            # Center score (inverse distance to center)
            dist_to_center = np.sqrt(
                (court.center[0] - frame_center[0]) ** 2 +
                (court.center[1] - frame_center[1]) ** 2
            )
            max_dist = np.sqrt(frame_center[0]**2 + frame_center[1]**2)
            center_score = 1 - (dist_to_center / max_dist)
            
            # Activity score
            if frame is not None:
                activity_score = self.calculate_activity_score(frame, court)
                court.activity_score = activity_score
            else:
                activity_score = court.activity_score
            
            # Combined score
            total_score = (
                size_weight * size_score +
                center_weight * center_score +
                activity_weight * activity_score
            )
            
            if total_score > best_score:
                best_score = total_score
                best_court = court
        
        if best_court:
            best_court.is_focused = True
            best_court.confidence = best_score
            
            # Update history for temporal consistency
            self._focused_court_history.append(best_court.bbox)
            if len(self._focused_court_history) > self._max_history:
                self._focused_court_history.pop(0)
        
        return best_court
    
    def smooth_court_detection(
        self, 
        current_court: Optional[CourtRegion]
    ) -> Optional[CourtRegion]:
        """
        Apply temporal smoothing to court detection.
        
        Uses history of detected courts to reduce jitter.
        
        Args:
            current_court: Currently detected court
            
        Returns:
            Smoothed court region
        """
        if not self._focused_court_history:
            return current_court
        
        if current_court is None:
            # Use average of history
            avg_bbox = tuple(
                int(np.mean([h[i] for h in self._focused_court_history]))
                for i in range(4)
            )
            return CourtRegion(
                bbox=avg_bbox,
                center=(avg_bbox[0] + avg_bbox[2]/2, avg_bbox[1] + avg_bbox[3]/2),
                is_focused=True
            )
        
        # Smooth with history using exponential moving average
        alpha = 0.7  # Weight for current detection
        
        history_bbox = [
            np.mean([h[i] for h in self._focused_court_history])
            for i in range(4)
        ]
        
        smoothed_bbox = tuple(
            int(alpha * current_court.bbox[i] + (1 - alpha) * history_bbox[i])
            for i in range(4)
        )
        
        smoothed = CourtRegion(
            bbox=smoothed_bbox,
            corners=current_court.corners,
            confidence=current_court.confidence,
            area=smoothed_bbox[2] * smoothed_bbox[3],
            center=(smoothed_bbox[0] + smoothed_bbox[2]/2, 
                   smoothed_bbox[1] + smoothed_bbox[3]/2),
            is_focused=True,
            activity_score=current_court.activity_score
        )
        
        return smoothed
    
    def detect(self, frame: np.ndarray) -> Optional[CourtRegion]:
        """
        Main detection method - detects and returns the focused court.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            The focused court region or None
        """
        # Method 1: Line-based detection
        lines, line_mask = self.line_detector.detect_lines(frame)
        line_courts = self.detect_court_from_lines(lines, frame.shape)
        
        # Method 2: Contour-based detection
        contours = self.detect_court_contours(frame)
        contour_courts = self.detect_court_from_contours(contours, frame.shape)
        
        # Combine detections
        all_courts = line_courts + contour_courts
        
        # Remove duplicates (overlapping courts)
        unique_courts = self._remove_overlapping_courts(all_courts)
        
        # Select focused court
        focused = self.select_focused_court(unique_courts, frame.shape, frame)
        
        # Apply temporal smoothing
        focused = self.smooth_court_detection(focused)
        
        return focused
    
    def _remove_overlapping_courts(
        self, 
        courts: List[CourtRegion],
        iou_threshold: float = 0.5
    ) -> List[CourtRegion]:
        """Remove overlapping court detections."""
        if not courts:
            return []
        
        # Sort by area (keep larger ones)
        courts = sorted(courts, key=lambda c: c.area, reverse=True)
        
        unique = []
        for court in courts:
            is_duplicate = False
            for existing in unique:
                iou = self._calculate_iou(court.bbox, existing.bbox)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(court)
        
        return unique
    
    def _calculate_iou(
        self, 
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
