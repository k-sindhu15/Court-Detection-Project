"""
Player Detector Module

Detects players in video frames using YOLO-based object detection.
Filters detections to only include players within the court region.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlayerDetection:
    """Represents a detected player."""
    
    # Bounding box (x1, y1, x2, y2)
    bbox: Tuple[int, int, int, int]
    
    # Confidence score
    confidence: float
    
    # Track ID (assigned by tracker)
    track_id: Optional[int] = None
    
    # Center point
    center: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    
    # Additional features
    features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Calculate center from bbox."""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return bbox in xyxy format."""
        return self.bbox
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return bbox in xywh format."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


class PlayerDetector:
    """
    Detects players using YOLO object detection.
    
    Supports YOLOv8 models with configurable confidence thresholds.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the player detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self._initialized = False
        
        # Detection parameters
        detection_config = self.config.get('player_detection', {})
        self.model_name = detection_config.get('model', 'yolov8n')
        self.confidence_threshold = detection_config.get('confidence_threshold', 0.5)
        self.iou_threshold = detection_config.get('iou_threshold', 0.45)
        self.classes = detection_config.get('classes', [0])  # 0 = person
        
        # Device selection
        processing_config = self.config.get('processing', {})
        self.device = processing_config.get('device', 'auto')
    
    def initialize(self) -> bool:
        """
        Initialize the YOLO model.
        
        Returns:
            True if initialization successful
        """
        try:
            from ultralytics import YOLO
            
            # Load model
            model_path = f"{self.model_name}.pt"
            logger.info(f"Loading YOLO model: {model_path}")
            
            self.model = YOLO(model_path, device='cpu')
            
            # Set device to CPU for compatibility
            self.device = 'cpu'
            
            logger.info(f"YOLO model loaded successfully on {self.device}")
            self._initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import ultralytics: {e}")
            logger.error("Please install ultralytics: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            return False
    
    def detect(
        self, 
        frame: np.ndarray,
        court_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> List[PlayerDetection]:
        """
        Detect players in frame.
        
        Args:
            frame: Input BGR frame
            court_bbox: Optional court bounding box to filter detections
            
        Returns:
            List of player detections
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for box in boxes:
                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                
                detection = PlayerDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence
                )
                
                # Filter by court region if provided
                if court_bbox is not None:
                    if self._is_in_court(detection, court_bbox):
                        detections.append(detection)
                else:
                    detections.append(detection)
        
        logger.debug(f"Detected {len(detections)} players")
        return detections
    
    def _is_in_court(
        self, 
        detection: PlayerDetection,
        court_bbox: Tuple[int, int, int, int],
        margin_ratio: float = 0.05
    ) -> bool:
        """
        Check if detection is within court boundaries.
        
        Args:
            detection: Player detection
            court_bbox: Court bounding box (x, y, w, h)
            margin_ratio: Margin around court
            
        Returns:
            True if player is in court
        """
        cx, cy = court_bbox[0], court_bbox[1]
        cw, ch = court_bbox[2], court_bbox[3]
        
        # Add margin
        margin_x = cw * margin_ratio
        margin_y = ch * margin_ratio
        
        court_x1 = cx - margin_x
        court_y1 = cy - margin_y
        court_x2 = cx + cw + margin_x
        court_y2 = cy + ch + margin_y
        
        # Check if player center is in court
        px, py = detection.center
        
        return (court_x1 <= px <= court_x2 and 
                court_y1 <= py <= court_y2)
    
    def visualize_detections(
        self, 
        frame: np.ndarray,
        detections: List[PlayerDetection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            color: Bounding box color
            thickness: Line thickness
            show_confidence: Show confidence score
            
        Returns:
            Frame with detections drawn
        """
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if det.track_id is not None:
                label = f"ID: {det.track_id}"
                if show_confidence:
                    label += f" ({det.confidence:.2f})"
            elif show_confidence:
                label = f"{det.confidence:.2f}"
            else:
                label = ""
            
            if label:
                # Background for text
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    result,
                    (x1, y1 - text_height - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                cv2.putText(
                    result,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return result
