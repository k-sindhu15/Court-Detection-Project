"""
Video Handler Module

Provides utilities for reading and writing video files.
Handles frame extraction and video reconstruction.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Container for video metadata."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    
    def __str__(self) -> str:
        return (
            f"VideoInfo(resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f}, frames={self.total_frames}, "
            f"duration={self.duration:.2f}s)"
        )


class VideoReader:
    """
    Video reader with frame-by-frame iteration support.
    
    Example:
        reader = VideoReader("input.mp4")
        for frame_idx, frame in reader:
            # Process frame
            pass
        reader.release()
    """
    
    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to input video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        self._info = self._extract_info()
        self._current_frame = 0
        
        logger.info(f"Opened video: {self._info}")
    
    def _extract_info(self) -> VideoInfo:
        """Extract video metadata."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        duration = total_frames / fps if fps > 0 else 0
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec
        )
    
    @property
    def info(self) -> VideoInfo:
        """Get video information."""
        return self._info
    
    @property
    def width(self) -> int:
        return self._info.width
    
    @property
    def height(self) -> int:
        return self._info.height
    
    @property
    def fps(self) -> float:
        return self._info.fps
    
    @property
    def total_frames(self) -> int:
        return self._info.total_frames
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.
        
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()
        if ret:
            self._current_frame += 1
        return ret, frame
    
    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read a specific frame by index.
        
        Args:
            frame_idx: Frame index to read
            
        Returns:
            Frame as numpy array or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        self._current_frame = frame_idx + 1 if ret else frame_idx
        return frame if ret else None
    
    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate over all frames."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._current_frame = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_idx = self._current_frame
            self._current_frame += 1
            yield frame_idx, frame
    
    def sample_frames(self, n_frames: int = 10) -> list:
        """
        Sample N frames evenly distributed across the video.
        
        Args:
            n_frames: Number of frames to sample
            
        Returns:
            List of (frame_idx, frame) tuples
        """
        indices = np.linspace(0, self.total_frames - 1, n_frames, dtype=int)
        frames = []
        
        for idx in indices:
            frame = self.read_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
        
        return frames
    
    def release(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Released video: {self.video_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """
    Video writer for creating output videos.
    
    Example:
        writer = VideoWriter("output.mp4", width=1920, height=1080, fps=30)
        writer.write(frame)
        writer.release()
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v"
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path for output video
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: FourCC codec code (default: mp4v)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise IOError(f"Failed to create video writer: {output_path}")
        
        self._frame_count = 0
        logger.info(f"Created video writer: {output_path} ({width}x{height} @ {fps}fps)")
    
    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.
        
        Args:
            frame: Frame to write (BGR format)
        """
        # Ensure frame matches expected dimensions
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
        self._frame_count += 1
    
    @property
    def frame_count(self) -> int:
        """Get number of frames written."""
        return self._frame_count
    
    def release(self) -> None:
        """Release video writer resources."""
        if self.writer is not None:
            self.writer.release()
            logger.info(f"Saved video: {self.output_path} ({self._frame_count} frames)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def create_writer_from_reader(
    reader: VideoReader,
    output_path: Union[str, Path],
    codec: str = "mp4v"
) -> VideoWriter:
    """
    Create a video writer with same properties as reader.
    
    Args:
        reader: Source video reader
        output_path: Path for output video
        codec: FourCC codec code
        
    Returns:
        VideoWriter instance
    """
    return VideoWriter(
        output_path=output_path,
        width=reader.width,
        height=reader.height,
        fps=reader.fps,
        codec=codec
    )
