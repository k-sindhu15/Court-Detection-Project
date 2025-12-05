"""
Player Tracker Module

Tracks detected players across frames maintaining consistent IDs.
Supports ByteTrack and other tracking algorithms.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from .player_detector import PlayerDetection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked player."""
    
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    
    # Track state
    is_confirmed: bool = False
    is_lost: bool = False
    time_since_update: int = 0
    
    # Trajectory history
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    max_trajectory_length: int = 30
    
    # Appearance features for re-identification
    features: Optional[np.ndarray] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update_trajectory(self):
        """Add current position to trajectory."""
        self.trajectory.append(self.center)
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)


class ByteTracker:
    """
    Implementation of ByteTrack algorithm for multi-object tracking.
    
    ByteTrack associates detections using both high and low confidence
    detections for robust tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ByteTracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # ByteTrack parameters
        bytetrack_config = self.config.get('player_tracking', {}).get('bytetrack', {})
        self.track_high_thresh = bytetrack_config.get('track_high_thresh', 0.5)
        self.track_low_thresh = bytetrack_config.get('track_low_thresh', 0.1)
        self.new_track_thresh = bytetrack_config.get('new_track_thresh', 0.6)
        self.track_buffer = bytetrack_config.get('track_buffer', 30)
        self.match_thresh = bytetrack_config.get('match_thresh', 0.8)
        
        # Track management
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
    
    def update(
        self, 
        detections: List[PlayerDetection]
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of player detections
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Separate high and low confidence detections
        high_detections = [d for d in detections if d.confidence >= self.track_high_thresh]
        low_detections = [d for d in detections if self.track_low_thresh <= d.confidence < self.track_high_thresh]
        
        # Get confirmed and unconfirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed]
        unconfirmed_tracks = [t for t in self.tracks if not t.is_confirmed]
        
        # First association: high confidence detections with confirmed tracks
        matched_tracks, unmatched_tracks, unmatched_detections = self._associate(
            confirmed_tracks, high_detections, self.match_thresh
        )
        
        # Update matched tracks
        for track, detection in matched_tracks:
            self._update_track(track, detection)
        
        # Second association: remaining tracks with low confidence detections
        matched_tracks2, unmatched_tracks2, _ = self._associate(
            unmatched_tracks, low_detections, self.match_thresh
        )
        
        for track, detection in matched_tracks2:
            self._update_track(track, detection)
        
        # Third association: unconfirmed tracks with remaining high confidence detections
        matched_tracks3, unmatched_unconfirmed, remaining_detections = self._associate(
            unconfirmed_tracks, unmatched_detections, self.match_thresh
        )
        
        for track, detection in matched_tracks3:
            self._update_track(track, detection)
            track.is_confirmed = True
        
        # Handle lost tracks
        for track in unmatched_tracks2:
            track.time_since_update += 1
            if track.time_since_update > self.track_buffer:
                track.is_lost = True
                self.lost_tracks.append(track)
            else:
                self.tracks.append(track)
        
        for track in unmatched_unconfirmed:
            track.time_since_update += 1
            if track.time_since_update > 3:  # Quick removal for unconfirmed
                track.is_lost = True
        
        # Create new tracks for unmatched high confidence detections
        for detection in remaining_detections:
            if detection.confidence >= self.new_track_thresh:
                new_track = self._create_track(detection)
                self.tracks.append(new_track)
        
        # Remove lost tracks from active tracks
        self.tracks = [t for t in self.tracks if not t.is_lost]
        
        # Clean up old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.track_buffer * 2]
        
        # Update trajectories
        for track in self.tracks:
            track.update_trajectory()
        
        return [t for t in self.tracks if t.is_confirmed]
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List[PlayerDetection],
        threshold: float
    ) -> Tuple[List[Tuple[Track, PlayerDetection]], List[Track], List[PlayerDetection]]:
        """
        Associate tracks with detections using IoU.
        
        Args:
            tracks: Existing tracks
            detections: New detections
            threshold: IoU threshold for matching
            
        Returns:
            Tuple of (matched pairs, unmatched tracks, unmatched detections)
        """
        if not tracks or not detections:
            return [], tracks.copy(), detections.copy()
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection.bbox)
        
        # Hungarian algorithm would be ideal here, using greedy for simplicity
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        while True:
            # Find best match
            if iou_matrix.size == 0 or np.max(iou_matrix) < threshold:
                break
            
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, det_idx = max_idx
            
            if iou_matrix[track_idx, det_idx] >= threshold:
                matched.append((tracks[track_idx], detections[det_idx]))
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                
                # Invalidate row and column
                iou_matrix[track_idx, :] = 0
                iou_matrix[:, det_idx] = 0
            else:
                break
        
        return (
            matched,
            [tracks[i] for i in unmatched_tracks],
            [detections[i] for i in unmatched_detections]
        )
    
    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_track(self, track: Track, detection: PlayerDetection) -> None:
        """Update track with new detection."""
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.time_since_update = 0
        if detection.features is not None:
            track.features = detection.features
    
    def _create_track(self, detection: PlayerDetection) -> Track:
        """Create a new track from detection."""
        track = Track(
            track_id=self.next_track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            is_confirmed=detection.confidence >= self.new_track_thresh,
            features=detection.features
        )
        self.next_track_id += 1
        return track
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks = []
        self.lost_tracks = []
        self.next_track_id = 1
        self.frame_count = 0


class PlayerTracker:
    """
    High-level player tracking interface.
    
    Combines detection and tracking for seamless player tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize player tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        from .player_detector import PlayerDetector
        self.detector = PlayerDetector(config)
        self.tracker = ByteTracker(config)
        
        # Visualization settings
        vis_config = self.config.get('player_tracking', {}).get('visualization', {})
        self.show_track_id = vis_config.get('show_track_id', True)
        self.show_trajectory = vis_config.get('show_trajectory', True)
        self.trajectory_length = vis_config.get('trajectory_length', 30)
        self.bbox_thickness = vis_config.get('bbox_thickness', 2)
        
        # Color palette for tracks
        self.colors = self._generate_colors(20)
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(
                np.array([[[hue, 255, 200]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR
            )[0, 0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def process_frame(
        self,
        frame: np.ndarray,
        court_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[List[Track], np.ndarray]:
        """
        Process a frame: detect and track players.
        
        Args:
            frame: Input BGR frame
            court_bbox: Optional court bounding box for filtering
            
        Returns:
            Tuple of (active tracks, annotated frame)
        """
        # Detect players
        detections = self.detector.detect(frame, court_bbox)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Annotate frame
        annotated = self.visualize_tracks(frame, tracks)
        
        return tracks, annotated
    
    def visualize_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track]
    ) -> np.ndarray:
        """
        Visualize tracks on frame.
        
        Args:
            frame: Input frame
            tracks: List of active tracks
            
        Returns:
            Annotated frame
        """
        result = frame.copy()
        
        for track in tracks:
            color = self.colors[track.track_id % len(self.colors)]
            x1, y1, x2, y2 = track.bbox
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.bbox_thickness)
            
            # Draw track ID
            if self.show_track_id:
                label = f"ID: {track.track_id}"
                cv2.putText(
                    result,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Draw trajectory
            if self.show_trajectory and len(track.trajectory) > 1:
                points = np.array(track.trajectory, dtype=np.int32)
                cv2.polylines(
                    result,
                    [points],
                    False,
                    color,
                    2
                )
        
        return result
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracker.reset()
