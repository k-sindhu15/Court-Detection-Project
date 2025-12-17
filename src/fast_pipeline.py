"""
Fast Processing Pipeline

Optimized for speed with:
- Frame skipping
- Reduced resolution processing
- Court detection caching
- Minimal per-frame operations
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class FastProcessor:
    """
    Ultra-fast video processor optimized for speed.
    
    Key optimizations:
    1. Process every Nth frame
    2. Detect court once and reuse
    3. Simple rectangular masking
    4. Optional player tracking
    """
    
    def __init__(
        self,
        frame_skip: int = 3,
        resize_factor: float = 1.0,
        enable_tracking: bool = True,
        mask_color: Tuple[int, int, int] = (0, 0, 0),
        confidence: float = 0.5,
        model: str = "yolov8n"
    ):
        """
        Initialize fast processor.
        
        Args:
            frame_skip: Process every Nth frame (1=all, 3=every 3rd)
            resize_factor: Resize factor (0.5 = half size)
            enable_tracking: Whether to track players
            mask_color: BGR color for masked regions
            confidence: YOLO confidence threshold
            model: YOLO model name
        """
        self.frame_skip = max(1, frame_skip)
        self.resize_factor = min(1.0, max(0.25, resize_factor))
        self.enable_tracking = enable_tracking
        self.mask_color = mask_color
        self.confidence = confidence
        self.model_name = model
        
        # Detection model (lazy load)
        self._yolo = None
        self._court_bbox = None  # Cached court detection
        
    def _load_yolo(self):
        """Load YOLO model."""
        if self._yolo is None and self.enable_tracking:
            try:
                from ultralytics import YOLO
                # Force CPU usage for compatibility with low-spec systems
                self._yolo = YOLO(f"{self.model_name}.pt", device='cpu')
                logger.info(f"Loaded {self.model_name} on CPU")
            except Exception as e:
                logger.warning(f"Failed to load YOLO: {e}")
                self.enable_tracking = False
    
    def detect_court_simple(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect badminton court boundaries.
        
        Strategy: Find the rectangular white court boundary lines.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Find white pixels (court lines are white)
        _, white_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        
        # Step 2: Find lines using Hough Transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=100, maxLineGap=20)
        
        if lines is not None and len(lines) > 5:
            # Separate horizontal and vertical lines
            h_lines = []  # Horizontal
            v_lines = []  # Vertical
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                if angle < 20 or angle > 160:  # Horizontal
                    h_lines.append((min(y1,y2), max(y1,y2), min(x1,x2), max(x1,x2)))
                elif 70 < angle < 110:  # Vertical
                    v_lines.append((min(x1,x2), max(x1,x2), min(y1,y2), max(y1,y2)))
            
            # Find court boundaries from lines
            if h_lines and v_lines:
                # Top line (lowest y value horizontal line in upper half)
                top_lines = [l for l in h_lines if l[0] < h * 0.4]
                top_y = min([l[0] for l in top_lines]) if top_lines else int(h * 0.05)
                
                # Bottom line (highest y value horizontal line in lower half)  
                bot_lines = [l for l in h_lines if l[1] > h * 0.6]
                bot_y = max([l[1] for l in bot_lines]) if bot_lines else int(h * 0.95)
                
                # Left line (lowest x value vertical line in left half)
                left_lines = [l for l in v_lines if l[0] < w * 0.4]
                left_x = min([l[0] for l in left_lines]) if left_lines else int(w * 0.1)
                
                # Right line (highest x value vertical line in right half)
                right_lines = [l for l in v_lines if l[1] > w * 0.6]
                right_x = max([l[1] for l in right_lines]) if right_lines else int(w * 0.9)
                
                # Validate and adjust
                cw = right_x - left_x
                ch = bot_y - top_y
                
                if cw > w * 0.3 and ch > h * 0.3:
                    # Add small padding
                    pad = 15
                    left_x = max(0, left_x - pad)
                    top_y = max(0, top_y - pad)
                    right_x = min(w, right_x + pad)
                    bot_y = min(h, bot_y + pad)
                    
                    logger.info(f"Court from lines: ({left_x},{top_y}) to ({right_x},{bot_y})")
                    return (left_x, top_y, right_x - left_x, bot_y - top_y)
        
        # Fallback: Use center 80% of frame
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.08)
        logger.warning(f"Using center crop fallback")
        return (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    
    def detect_players(self, frame: np.ndarray, court_bbox: Tuple[int, int, int, int], court_mask: np.ndarray = None) -> list:
        """
        Detect players using YOLO - only those INSIDE the court mask.
        
        Args:
            frame: Input frame
            court_bbox: (x, y, width, height) of court bounding box
            court_mask: Binary mask where white=inside court, black=outside
        """
        if self._yolo is None:
            return []
        
        results = self._yolo(
            frame,
            conf=self.confidence,
            classes=[0],  # person only
            verbose=False
        )
        
        players = []
        cx, cy, cw, ch = court_bbox
        
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Get player's foot position (bottom center of bbox)
                foot_x = (x1 + x2) // 2
                foot_y = y2  # Bottom of bounding box (feet)
                
                # Check if within court bounding box first
                if not (cx <= foot_x <= cx + cw and cy <= foot_y <= cy + ch):
                    continue
                
                # If we have a court mask, check if player's feet are inside the court lines
                if court_mask is not None:
                    # Convert to mask coordinates (relative to court bbox)
                    mask_x = foot_x - cx
                    mask_y = foot_y - cy
                    
                    # Ensure within mask bounds
                    if 0 <= mask_x < court_mask.shape[1] and 0 <= mask_y < court_mask.shape[0]:
                        # Only include if feet are on the court (white area in mask)
                        if court_mask[mask_y, mask_x] == 0:
                            continue  # Skip - player is outside court lines (referee)
                
                players.append((x1, y1, x2, y2, float(box.conf[0])))
        
        return players
    
    def apply_mask(
        self, 
        frame: np.ndarray, 
        court_bbox: Tuple[int, int, int, int],
        crop: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mask to hide areas outside the court lines.
        
        Args:
            frame: Input frame
            court_bbox: (x, y, width, height) of court region
            crop: If True, crop frame to court region
            
        Returns:
            Tuple of (processed frame, court mask for the cropped region)
        """
        h, w = frame.shape[:2]
        x, y, cw, ch = court_bbox
        
        # Ensure bounds are valid
        x = max(0, x)
        y = max(0, y)
        cw = min(cw, w - x)
        ch = min(ch, h - y)
        
        # First crop to the court bounding box
        cropped = frame[y:y+ch, x:x+cw].copy()
        
        # Now create a mask for the area INSIDE the white court lines
        # Detect white lines in the cropped region
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Find white pixels (court lines)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Dilate to connect lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_dilated = cv2.dilate(white_mask, kernel, iterations=3)
        
        # Find contours of the court boundary
        contours, _ = cv2.findContours(white_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask - start with black (everything masked)
        court_mask = np.zeros((ch, cw), dtype=np.uint8)
        
        if contours:
            # Find the largest contour (should be the outer court boundary)
            largest = max(contours, key=cv2.contourArea)
            
            # Fill the court area with white (unmask the court interior)
            cv2.fillPoly(court_mask, [largest], 255)
            
            # Also try convex hull for better coverage
            hull = cv2.convexHull(largest)
            cv2.fillPoly(court_mask, [hull], 255)
        else:
            # Fallback: use the entire cropped region
            court_mask[:] = 255
        
        # Apply mask - black out areas outside the court
        masked = cropped.copy()
        masked[court_mask == 0] = self.mask_color
        
        return masked, court_mask
    
    def draw_players(self, frame: np.ndarray, players: list) -> np.ndarray:
        """Draw player bounding boxes."""
        for i, (x1, y1, x2, y2, conf) in enumerate(players):
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"P{i+1}"
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> tuple:
        """
        Process video with optimizations.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Optional callback(current, total)
            
        Returns:
            Tuple of (success: bool, actual_output_path: str or None)
        """
        logger.info(f"Fast processing: {input_path}")
        logger.info(f"Settings: skip={self.frame_skip}, resize={self.resize_factor}, tracking={self.enable_tracking}")
        
        # Load YOLO if needed
        if self.enable_tracking:
            self._load_yolo()
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {input_path}")
            return False, None
        
        # Get properties
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate resize dimensions
        resize_w = int(orig_w * self.resize_factor)
        resize_h = int(orig_h * self.resize_factor)
        out_fps = fps / self.frame_skip
        
        logger.info(f"Input: {orig_w}x{orig_h} @ {fps}fps, {total_frames} frames")
        
        # FIRST: Detect court from first frame to determine crop size
        ret, first_frame = cap.read()
        if not ret:
            logger.error("Cannot read first frame")
            cap.release()
            return False, None
        
        # Resize to working size for court detection
        if self.resize_factor < 1.0:
            first_frame = cv2.resize(first_frame, (resize_w, resize_h))
        
        # Use manual bbox if provided, otherwise auto-detect
        if hasattr(self, '_manual_court_bbox') and self._manual_court_bbox:
            # Scale manual bbox by resize factor
            mx, my, mw, mh = self._manual_court_bbox
            self._court_bbox = (
                int(mx * self.resize_factor),
                int(my * self.resize_factor),
                int(mw * self.resize_factor),
                int(mh * self.resize_factor)
            )
            logger.info(f"Using manual court bbox (scaled): {self._court_bbox}")
        else:
            self._court_bbox = self.detect_court_simple(first_frame)
            logger.info(f"Auto-detected court: {self._court_bbox}")
        
        cx, cy, cw, ch = self._court_bbox
        logger.info(f"Court detected: x={cx}, y={cy}, w={cw}, h={ch}")
        
        # OUTPUT SIZE = CROPPED COURT SIZE
        out_w, out_h = cw, ch
        logger.info(f"Output (cropped): {out_w}x{out_h} @ {out_fps:.1f}fps")
        
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create output using imageio (better codec support)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use imageio for MP4 output (browser compatible)
        import imageio
        output_mp4 = output_path if output_path.endswith('.mp4') else output_path.replace('.avi', '.mp4')
        
        try:
            writer = imageio.get_writer(
                output_mp4, 
                fps=max(1.0, out_fps),
                codec='libx264',
                quality=7,
                pixelformat='yuv420p',
                macro_block_size=None
            )
            use_imageio = True
            logger.info(f"Using imageio with libx264: {output_mp4}")
        except Exception as e:
            logger.warning(f"imageio failed: {e}, falling back to OpenCV")
            use_imageio = False
            output_avi = output_path.replace('.mp4', '.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_avi, fourcc, max(1.0, out_fps), (out_w, out_h))
            output_mp4 = output_avi
            
            if not writer.isOpened():
                logger.error(f"Cannot create output video")
                cap.release()
                return False, None
        
        self._output_path = output_mp4
        self._use_imageio = use_imageio
        
        frame_idx = 0
        processed = 0
        
        # Detect court mask from first frame (only once)
        self._court_mask = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Skip frames
                if self.frame_skip > 1 and (frame_idx - 1) % self.frame_skip != 0:
                    continue
                
                # Resize to working size
                if self.resize_factor < 1.0:
                    frame = cv2.resize(frame, (resize_w, resize_h))
                
                # CROP and MASK to court region FIRST
                if self._court_bbox:
                    frame, mask = self.apply_mask(frame, self._court_bbox, crop=True)
                    
                    # Store mask from first processed frame
                    if self._court_mask is None:
                        self._court_mask = mask
                
                # Detect players AFTER cropping - use mask to filter out referees
                players = []
                if self.enable_tracking and self._yolo is not None and self._court_mask is not None:
                    # Detect on cropped frame, use (0,0,w,h) as bbox since already cropped
                    h_crop, w_crop = frame.shape[:2]
                    players = self.detect_players(frame, (0, 0, w_crop, h_crop), self._court_mask)
                    
                    # Draw players on cropped frame (no coordinate adjustment needed)
                    frame = self.draw_players(frame, players)
                
                # Write frame (imageio needs RGB, OpenCV uses BGR)
                if self._use_imageio:
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    writer.write(frame)
                processed += 1
                
                # Progress callback
                if progress_callback and frame_idx % 10 == 0:
                    progress_callback(frame_idx, total_frames)
                
                # Log progress
                if processed % 100 == 0:
                    pct = (frame_idx / total_frames) * 100
                    logger.info(f"Progress: {pct:.1f}%")
            
            logger.info(f"Done! Processed {processed} frames (skipped {frame_idx - processed})")
            cap.release()
            if self._use_imageio:
                writer.close()
            else:
                writer.release()
            return True, output_mp4
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            cap.release()
            try:
                if self._use_imageio:
                    writer.close()
                else:
                    writer.release()
            except:
                pass
            return False, None
