"""
Main Processing Pipeline

Integrates court detection, masking, and player tracking into a
complete video processing pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator, Callable
from dataclasses import dataclass
import logging
from tqdm import tqdm

from .video_processing.video_handler import VideoReader, VideoWriter, create_writer_from_reader
from .court_detection.court_detector import CourtRegionDetector, CourtRegion
from .court_detection.court_masker import CourtMasker, AdjacentCourtMasker
from .player_tracking.tracker import PlayerTracker, Track
from .utils.config import ConfigManager, get_config

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Container for processing results."""
    
    # Court detection
    court: Optional[CourtRegion] = None
    
    # Player tracks
    tracks: list = None
    
    # Processed frame
    frame: Optional[np.ndarray] = None
    
    # Frame index
    frame_idx: int = 0
    
    def __post_init__(self):
        if self.tracks is None:
            self.tracks = []


class BadmintonCourtProcessor:
    """
    Main processing pipeline for badminton court detection and player tracking.
    
    This class orchestrates:
    1. Video reading
    2. Court detection and selection
    3. Court masking
    4. Player detection and tracking
    5. Video output
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the processor.
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (overrides file)
        """
        # Load configuration
        self.config_manager = get_config(config_path)
        if config:
            for key, value in config.items():
                self.config_manager.set(key, value)
        
        self.config = self.config_manager.to_dict()
        
        # Initialize components (lazy loading)
        self._court_detector: Optional[CourtRegionDetector] = None
        self._court_masker: Optional[CourtMasker] = None
        self._player_tracker: Optional[PlayerTracker] = None
        
        # Processing state
        self._calibrated = False
        self._current_court: Optional[CourtRegion] = None
        
        logger.info("BadmintonCourtProcessor initialized")
    
    @property
    def court_detector(self) -> CourtRegionDetector:
        """Get or create court detector."""
        if self._court_detector is None:
            self._court_detector = CourtRegionDetector(self.config)
        return self._court_detector
    
    @property
    def court_masker(self) -> CourtMasker:
        """Get or create court masker."""
        if self._court_masker is None:
            self._court_masker = CourtMasker(self.config)
        return self._court_masker
    
    @property
    def player_tracker(self) -> PlayerTracker:
        """Get or create player tracker."""
        if self._player_tracker is None:
            self._player_tracker = PlayerTracker(self.config)
        return self._player_tracker
    
    def calibrate(self, video_path: str, n_frames: int = 30) -> None:
        """
        Calibrate the processor using sample frames from video.
        
        Args:
            video_path: Path to input video
            n_frames: Number of frames to use for calibration
        """
        logger.info(f"Calibrating from video: {video_path}")
        
        with VideoReader(video_path) as reader:
            # Update config with video properties
            self.config_manager.update_for_video(
                reader.width, reader.height, reader.fps
            )
            
            # Sample frames
            samples = reader.sample_frames(n_frames)
            frames = [frame for _, frame in samples]
            
            # Calibrate court detector
            self.court_detector.calibrate(frames)
        
        self._calibrated = True
        logger.info("Calibration complete")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        mask_court: bool = True,
        track_players: bool = True
    ) -> ProcessingResult:
        """
        Process a single frame.
        
        Args:
            frame: Input BGR frame
            frame_idx: Frame index
            mask_court: Whether to apply court masking
            track_players: Whether to track players
            
        Returns:
            ProcessingResult with court, tracks, and processed frame
        """
        result = ProcessingResult(frame_idx=frame_idx)
        processed_frame = frame.copy()
        
        # Detect court
        court = self.court_detector.detect(frame)
        result.court = court
        self._current_court = court
        
        if court is None:
            logger.warning(f"No court detected in frame {frame_idx}")
            result.frame = processed_frame
            return result
        
        # Track players
        if track_players:
            court_bbox = court.bbox
            tracks, _ = self.player_tracker.process_frame(frame, court_bbox)
            result.tracks = tracks
        
        # Apply masking
        if mask_court:
            processed_frame = self.court_masker.mask_frame(
                processed_frame, 
                court,
                draw_boundary=self.config.get('output', {}).get('show_court_boundary', False)
            )
        
        # Draw player tracks on masked frame
        if track_players and result.tracks:
            processed_frame = self.player_tracker.visualize_tracks(
                processed_frame, result.tracks
            )
        
        result.frame = processed_frame
        return result
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        mask_court: bool = True,
        track_players: bool = True,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Process a complete video file.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            mask_court: Whether to apply court masking
            track_players: Whether to track players
            show_progress: Whether to show progress bar
            progress_callback: Optional callback(current, total)
        """
        logger.info(f"Processing video: {input_path}")
        logger.info(f"Output: {output_path}")
        
        # Calibrate if not already done
        if not self._calibrated:
            self.calibrate(input_path)
        
        # Open video
        with VideoReader(input_path) as reader:
            # Create writer with same properties
            with VideoWriter(
                output_path,
                reader.width,
                reader.height,
                reader.fps,
                codec=self.config.get('video', {}).get('output_codec', 'mp4v')
            ) as writer:
                
                # Process frames
                iterator = reader
                if show_progress:
                    iterator = tqdm(
                        reader,
                        total=reader.total_frames,
                        desc="Processing",
                        unit="frame"
                    )
                
                for frame_idx, frame in iterator:
                    # Process frame
                    result = self.process_frame(
                        frame,
                        frame_idx=frame_idx,
                        mask_court=mask_court,
                        track_players=track_players
                    )
                    
                    # Write output frame
                    if result.frame is not None:
                        writer.write(result.frame)
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(frame_idx + 1, reader.total_frames)
        
        logger.info(f"Video processing complete: {output_path}")
    
    def process_video_generator(
        self,
        input_path: str,
        mask_court: bool = True,
        track_players: bool = True
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process video yielding results frame by frame.
        
        Useful for real-time display or custom output handling.
        
        Args:
            input_path: Path to input video
            mask_court: Whether to apply court masking
            track_players: Whether to track players
            
        Yields:
            ProcessingResult for each frame
        """
        if not self._calibrated:
            self.calibrate(input_path)
        
        with VideoReader(input_path) as reader:
            for frame_idx, frame in reader:
                result = self.process_frame(
                    frame,
                    frame_idx=frame_idx,
                    mask_court=mask_court,
                    track_players=track_players
                )
                yield result
    
    def reset(self) -> None:
        """Reset processor state for new video."""
        if self._player_tracker:
            self._player_tracker.reset()
        self._current_court = None
        self._calibrated = False


def process_video(
    input_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    mask_court: bool = True,
    track_players: bool = True,
    **kwargs
) -> None:
    """
    Convenience function to process a video file.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        config_path: Path to configuration file
        mask_court: Whether to apply court masking
        track_players: Whether to track players
        **kwargs: Additional configuration overrides
    """
    processor = BadmintonCourtProcessor(
        config_path=config_path,
        config=kwargs
    )
    
    processor.process_video(
        input_path,
        output_path,
        mask_court=mask_court,
        track_players=track_players
    )
