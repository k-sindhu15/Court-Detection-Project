"""
Court Masker Module

Applies masking to video frames to isolate the focused court.
Masks irrelevant regions (adjacent courts, background) with configurable colors.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging
from .court_detector import CourtRegion

logger = logging.getLogger(__name__)


@dataclass
class MaskConfig:
    """Configuration for court masking."""
    
    # Background color for masked regions (BGR)
    background_color: Tuple[int, int, int] = (0, 0, 0)
    
    # Feather/smooth edges
    feather_edges: bool = True
    feather_radius: int = 5
    
    # Expand mask slightly to ensure full court capture
    expand_ratio: float = 0.02
    
    # Mask behind the court (perspective-based)
    mask_behind_court: bool = True
    
    # Show mask boundary for debugging
    show_boundary: bool = False
    boundary_color: Tuple[int, int, int] = (0, 255, 0)
    boundary_thickness: int = 2


class CourtMasker:
    """
    Applies masking to isolate the focused court region.
    
    Supports:
    - Simple rectangular masking
    - Perspective-aware masking
    - Smooth edge transitions
    - Behind-court masking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the court masker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.mask_config = self._load_mask_config()
        
        # Cache for mask optimization
        self._cached_mask: Optional[np.ndarray] = None
        self._cached_region: Optional[Tuple[int, int, int, int]] = None
    
    def _load_mask_config(self) -> MaskConfig:
        """Load masking configuration."""
        masking_config = self.config.get('court_masking', {})
        
        return MaskConfig(
            background_color=tuple(masking_config.get('background_color', [0, 0, 0])),
            feather_edges=masking_config.get('feather_edges', True),
            feather_radius=masking_config.get('feather_radius', 5),
            expand_ratio=masking_config.get('expand_mask_ratio', 0.02),
            mask_behind_court=masking_config.get('mask_behind_court', True),
            show_boundary=self.config.get('output', {}).get('show_court_boundary', False)
        )
    
    def create_rectangular_mask(
        self, 
        frame_shape: Tuple[int, int, int],
        court: CourtRegion,
        expand: bool = True
    ) -> np.ndarray:
        """
        Create a rectangular mask for the court region.
        
        Args:
            frame_shape: Shape of the frame (height, width, channels)
            court: Court region to mask
            expand: Whether to expand the mask slightly
            
        Returns:
            Binary mask (255 = court, 0 = masked area)
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        x, y, w, h = court.bbox
        
        # Expand region if requested
        if expand:
            expand_x = int(w * self.mask_config.expand_ratio)
            expand_y = int(h * self.mask_config.expand_ratio)
            
            x = max(0, x - expand_x)
            y = max(0, y - expand_y)
            w = min(width - x, w + 2 * expand_x)
            h = min(height - y, h + 2 * expand_y)
        
        # Draw rectangle on mask
        mask[y:y+h, x:x+w] = 255
        
        return mask
    
    def create_polygon_mask(
        self, 
        frame_shape: Tuple[int, int, int],
        court: CourtRegion
    ) -> np.ndarray:
        """
        Create a polygon mask using court corners.
        
        More accurate than rectangular for perspective-distorted courts.
        
        Args:
            frame_shape: Shape of the frame
            court: Court region with corners
            
        Returns:
            Binary mask
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if court.corners is not None:
            # Expand corners slightly
            center = np.mean(court.corners, axis=0)
            expanded_corners = center + (court.corners - center) * (1 + self.mask_config.expand_ratio)
            expanded_corners = expanded_corners.astype(np.int32)
            
            cv2.fillPoly(mask, [expanded_corners], 255)
        else:
            # Fall back to rectangular
            return self.create_rectangular_mask(frame_shape, court)
        
        return mask
    
    def create_behind_court_mask(
        self, 
        frame_shape: Tuple[int, int, int],
        court: CourtRegion
    ) -> np.ndarray:
        """
        Create a mask that includes the court but masks behind it.
        
        Uses perspective analysis to determine the "back" of the court.
        
        Args:
            frame_shape: Shape of the frame
            court: Court region
            
        Returns:
            Binary mask
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        x, y, w, h = court.bbox
        
        # Determine if this is a top-down or side view
        aspect_ratio = w / h if h > 0 else 1
        
        if aspect_ratio > 1.5:
            # Wide view (top-down or similar)
            # Mask above and below the court
            mask[y:y+h, :] = 255
        else:
            # Tall view (side angle)
            # Create a trapezoid mask (perspective)
            # The back of the court appears smaller due to perspective
            
            # Estimate perspective shrink factor
            shrink_factor = 0.85
            
            # Create trapezoid points
            top_left = (x + int(w * (1 - shrink_factor) / 2), y)
            top_right = (x + w - int(w * (1 - shrink_factor) / 2), y)
            bottom_left = (x, y + h)
            bottom_right = (x + w, y + h)
            
            points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def feather_mask(self, mask: np.ndarray, radius: Optional[int] = None) -> np.ndarray:
        """
        Apply Gaussian blur to mask edges for smooth transitions.
        
        Args:
            mask: Binary mask
            radius: Blur radius (uses config if not specified)
            
        Returns:
            Feathered mask with smooth edges
        """
        if radius is None:
            radius = self.mask_config.feather_radius
        
        if radius <= 0:
            return mask
        
        # Ensure radius is odd
        radius = radius if radius % 2 == 1 else radius + 1
        
        # Apply Gaussian blur for smooth edges
        feathered = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)
        
        return feathered
    
    def apply_mask(
        self, 
        frame: np.ndarray, 
        mask: np.ndarray,
        invert: bool = False
    ) -> np.ndarray:
        """
        Apply mask to frame, filling masked areas with background color.
        
        Args:
            frame: Input BGR frame
            mask: Binary or feathered mask
            invert: If True, mask the court and show surroundings
            
        Returns:
            Masked frame
        """
        if invert:
            mask = 255 - mask
        
        # Normalize mask to 0-1 range
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # Create 3-channel mask
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        
        # Create background
        background = np.full_like(frame, self.mask_config.background_color, dtype=np.uint8)
        
        # Blend frame and background based on mask
        result = (frame.astype(np.float32) * mask_3ch + 
                  background.astype(np.float32) * (1 - mask_3ch))
        
        return result.astype(np.uint8)
    
    def create_court_mask(
        self, 
        frame: np.ndarray,
        court: CourtRegion,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Create the complete court mask with all configured options.
        
        Args:
            frame: Input frame
            court: Court region
            use_cache: Whether to use cached mask if region unchanged
            
        Returns:
            Final mask for the court
        """
        # Check cache
        if use_cache and self._cached_mask is not None:
            if self._cached_region == court.bbox:
                return self._cached_mask
        
        frame_shape = frame.shape
        
        # Create base mask
        if court.corners is not None and len(court.corners) == 4:
            mask = self.create_polygon_mask(frame_shape, court)
        elif self.mask_config.mask_behind_court:
            mask = self.create_behind_court_mask(frame_shape, court)
        else:
            mask = self.create_rectangular_mask(frame_shape, court)
        
        # Apply feathering
        if self.mask_config.feather_edges:
            mask = self.feather_mask(mask)
        
        # Cache the mask
        self._cached_mask = mask
        self._cached_region = court.bbox
        
        return mask
    
    def mask_frame(
        self, 
        frame: np.ndarray, 
        court: CourtRegion,
        draw_boundary: bool = False
    ) -> np.ndarray:
        """
        Apply complete masking to a frame.
        
        Main method for masking - creates mask and applies it.
        
        Args:
            frame: Input BGR frame
            court: Court region to preserve
            draw_boundary: Whether to draw court boundary
            
        Returns:
            Masked frame
        """
        # Create mask
        mask = self.create_court_mask(frame, court)
        
        # Apply mask
        result = self.apply_mask(frame, mask)
        
        # Draw boundary if requested
        if draw_boundary or self.mask_config.show_boundary:
            result = self._draw_boundary(result, court)
        
        return result
    
    def _draw_boundary(self, frame: np.ndarray, court: CourtRegion) -> np.ndarray:
        """Draw court boundary on frame."""
        result = frame.copy()
        
        if court.corners is not None:
            cv2.polylines(
                result, 
                [court.corners.astype(np.int32)], 
                True,
                self.mask_config.boundary_color,
                self.mask_config.boundary_thickness
            )
        else:
            x, y, w, h = court.bbox
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                self.mask_config.boundary_color,
                self.mask_config.boundary_thickness
            )
        
        return result
    
    def crop_to_court(
        self, 
        frame: np.ndarray, 
        court: CourtRegion,
        padding_ratio: float = 0.02
    ) -> np.ndarray:
        """
        Crop frame to court region with optional padding.
        
        Args:
            frame: Input frame
            court: Court region
            padding_ratio: Padding as ratio of court size
            
        Returns:
            Cropped frame
        """
        x, y, w, h = court.bbox
        height, width = frame.shape[:2]
        
        # Add padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        
        return frame[y1:y2, x1:x2].copy()
    
    def mask_and_crop(
        self, 
        frame: np.ndarray, 
        court: CourtRegion,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Apply masking and crop to court region.
        
        Args:
            frame: Input frame
            court: Court region
            target_size: Optional (width, height) to resize to
            
        Returns:
            Masked and cropped frame
        """
        # First apply mask
        masked = self.mask_frame(frame, court)
        
        # Then crop
        cropped = self.crop_to_court(masked, court)
        
        # Resize if target size specified
        if target_size is not None:
            cropped = cv2.resize(cropped, target_size)
        
        return cropped


class AdjacentCourtMasker:
    """
    Specialized masker for handling adjacent courts.
    
    Detects and masks courts that are adjacent to the focused court
    while preserving the main court area.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the adjacent court masker."""
        self.config = config or {}
        self.base_masker = CourtMasker(config)
    
    def detect_adjacent_courts(
        self, 
        frame: np.ndarray,
        focused_court: CourtRegion,
        all_courts: list
    ) -> list:
        """
        Identify courts that are adjacent to the focused court.
        
        Args:
            frame: Current frame
            focused_court: The main court
            all_courts: All detected courts
            
        Returns:
            List of adjacent courts
        """
        adjacent = []
        
        for court in all_courts:
            if court is focused_court or court.is_focused:
                continue
            
            # Check if adjacent (sharing edge or nearby)
            if self._is_adjacent(focused_court.bbox, court.bbox):
                adjacent.append(court)
        
        return adjacent
    
    def _is_adjacent(
        self, 
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        threshold_ratio: float = 0.1
    ) -> bool:
        """Check if two bounding boxes are adjacent."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate gap between boxes
        gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        
        # Threshold based on court size
        threshold = max(w1, h1, w2, h2) * threshold_ratio
        
        return gap_x < threshold and gap_y < threshold
    
    def mask_adjacent_courts(
        self, 
        frame: np.ndarray,
        focused_court: CourtRegion,
        adjacent_courts: list
    ) -> np.ndarray:
        """
        Mask adjacent courts while preserving the focused court.
        
        Args:
            frame: Input frame
            focused_court: Main court to preserve
            adjacent_courts: Courts to mask
            
        Returns:
            Frame with adjacent courts masked
        """
        result = frame.copy()
        
        for court in adjacent_courts:
            # Create mask for this court
            mask = self.base_masker.create_rectangular_mask(
                frame.shape, court, expand=True
            )
            
            # Apply mask to result
            result = self.base_masker.apply_mask(result, mask, invert=True)
        
        return result
