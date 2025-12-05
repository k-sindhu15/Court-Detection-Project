#!/usr/bin/env python3
"""
Badminton Court Detection - Command Line Interface

Main entry point for the badminton court detection and player tracking system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Badminton Court Detection and Player Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage - process video with default settings
  python main.py input.mp4 output.mp4

  # Process with player tracking disabled
  python main.py input.mp4 output.mp4 --no-tracking

  # Process with custom configuration
  python main.py input.mp4 output.mp4 --config config/custom.yaml

  # Mask-only mode (no player tracking)
  python main.py input.mp4 output.mp4 --mask-only

  # Show processing preview
  python main.py input.mp4 output.mp4 --preview
        '''
    )
    
    # Positional arguments
    parser.add_argument(
        'input',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Path for output video file'
    )
    
    # Optional arguments
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--no-mask',
        action='store_true',
        help='Disable court masking'
    )
    parser.add_argument(
        '--no-tracking',
        action='store_true',
        help='Disable player tracking'
    )
    parser.add_argument(
        '--mask-only',
        action='store_true',
        help='Only apply masking (equivalent to --no-tracking)'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show processing preview window'
    )
    parser.add_argument(
        '--show-boundary',
        action='store_true',
        help='Draw court boundary on output'
    )
    parser.add_argument(
        '--bg-color',
        type=str,
        default='0,0,0',
        help='Mask background color as R,G,B (default: 0,0,0 = black)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='YOLO model for player detection (default: yolov8n)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with frame-by-frame output'
    )
    
    return parser


def validate_input(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return False
    
    # Check input is a video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    if input_path.suffix.lower() not in video_extensions:
        logger.warning(f"Input file may not be a video: {input_path.suffix}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {args.config}")
            return False
    
    # Validate background color
    try:
        parts = args.bg_color.split(',')
        if len(parts) != 3:
            raise ValueError("Need exactly 3 values")
        for p in parts:
            val = int(p)
            if not 0 <= val <= 255:
                raise ValueError(f"Value out of range: {val}")
    except ValueError as e:
        logger.error(f"Invalid background color format: {e}")
        return False
    
    return True


def parse_bg_color(color_str: str) -> tuple:
    """Parse background color string to tuple."""
    parts = color_str.split(',')
    # Return as BGR for OpenCV
    return (int(parts[2]), int(parts[1]), int(parts[0]))


def process_with_preview(
    input_path: str,
    output_path: str,
    config: dict,
    mask_court: bool = True,
    track_players: bool = True
) -> None:
    """Process video with preview window."""
    import cv2
    from src.pipeline import BadmintonCourtProcessor
    
    processor = BadmintonCourtProcessor(config=config)
    
    # Open video for preview
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calibrate
    processor.calibrate(input_path)
    
    # Create preview window
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preview', 1280, 720)
    
    frame_idx = 0
    paused = False
    
    print("\nControls:")
    print("  Space: Pause/Resume")
    print("  Q/Esc: Quit")
    print("  S: Save current frame")
    print()
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = processor.process_frame(
                frame,
                frame_idx=frame_idx,
                mask_court=mask_court,
                track_players=track_players
            )
            
            # Write to output
            if result.frame is not None:
                out.write(result.frame)
            
            # Show preview
            preview_frame = result.frame if result.frame is not None else frame
            
            # Add progress info
            progress = f"Frame: {frame_idx}/{total_frames}"
            cv2.putText(preview_frame, progress, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Preview', preview_frame)
            frame_idx += 1
        
        # Handle key presses
        key = cv2.waitKey(1 if not paused else 100) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or Escape
            print("\nProcessing interrupted by user")
            break
        elif key == ord(' '):  # Space
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('s'):  # Save frame
            save_path = f"frame_{frame_idx:06d}.png"
            cv2.imwrite(save_path, preview_frame)
            print(f"Saved: {save_path}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nOutput saved to: {output_path}")


def main():
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose or args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not validate_input(args):
        sys.exit(1)
    
    # Build configuration
    config = {}
    
    # Background color
    bg_color = parse_bg_color(args.bg_color)
    config['court_masking'] = {'background_color': list(bg_color)}
    
    # Show boundary
    if args.show_boundary:
        config['output'] = {'show_court_boundary': True}
    
    # Player detection model
    config['player_detection'] = {
        'model': args.model,
        'confidence_threshold': args.confidence
    }
    
    # Determine processing options
    mask_court = not args.no_mask
    track_players = not (args.no_tracking or args.mask_only)
    
    logger.info(f"Processing: {args.input} -> {args.output}")
    logger.info(f"Options: mask={mask_court}, tracking={track_players}")
    
    try:
        if args.preview:
            # Process with preview window
            process_with_preview(
                args.input,
                args.output,
                config,
                mask_court=mask_court,
                track_players=track_players
            )
        else:
            # Standard processing
            from src.pipeline import process_video
            
            process_video(
                args.input,
                args.output,
                config_path=args.config,
                mask_court=mask_court,
                track_players=track_players,
                **config
            )
        
        logger.info("Processing complete!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
