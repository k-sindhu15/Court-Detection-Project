#!/usr/bin/env python3
"""
Badminton Court Detection - Web UI

A simple Gradio-based web interface for uploading and processing
badminton videos with court detection and player tracking.
"""

import gradio as gr
import tempfile
import os
import shutil
from pathlib import Path

# Processing options
YOLO_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]


def process_video(
    video_file,
    enable_masking: bool,
    enable_tracking: bool,
    show_boundary: bool,
    model_name: str,
    confidence: float,
    bg_color: str,
    progress=gr.Progress()
):
    """
    Process uploaded video with court detection and player tracking.
    
    Args:
        video_file: Uploaded video file path
        enable_masking: Whether to mask non-court regions
        enable_tracking: Whether to track players
        show_boundary: Whether to show court boundary
        model_name: YOLO model to use
        confidence: Detection confidence threshold
        bg_color: Background color for masked regions
        progress: Gradio progress tracker
    
    Returns:
        Path to processed video
    """
    if video_file is None:
        return None, "Please upload a video file."
    
    try:
        progress(0.1, desc="Initializing...")
        
        # Import processing modules
        from src.pipeline import BadmintonCourtProcessor
        
        # Parse background color
        try:
            r, g, b = map(int, bg_color.split(','))
            bg_color_bgr = [b, g, r]  # Convert RGB to BGR
        except:
            bg_color_bgr = [0, 0, 0]  # Default black
        
        # Build configuration
        config = {
            'player_detection': {
                'model': model_name,
                'confidence_threshold': confidence
            },
            'court_masking': {
                'background_color': bg_color_bgr
            },
            'output': {
                'show_court_boundary': show_boundary
            }
        }
        
        progress(0.2, desc="Loading models...")
        
        # Initialize processor
        processor = BadmintonCourtProcessor(config=config)
        
        # Create output path
        output_dir = tempfile.mkdtemp()
        input_path = video_file
        output_filename = f"processed_{Path(video_file).stem}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        progress(0.3, desc="Processing video...")
        
        # Process video
        processor.process_video(
            input_path=input_path,
            output_path=output_path,
            mask_court=enable_masking,
            track_players=enable_tracking,
            show_progress=False
        )
        
        progress(1.0, desc="Complete!")
        
        return output_path, f"✅ Processing complete! Video saved."
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def create_ui():
    """Create the Gradio web interface."""
    
    with gr.Blocks(title="Badminton Court Detection") as app:
        
        gr.Markdown(
            """
            # 🏸 Badminton Court Detection & Player Tracking
            
            Upload a badminton match video to automatically detect and isolate the focused court,
            mask irrelevant regions, and track players with consistent IDs.
            """
        )
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Video")
                
                video_input = gr.Video(
                    label="Input Video",
                    sources=["upload"],
                )
                
                gr.Markdown("### ⚙️ Processing Options")
                
                with gr.Group():
                    enable_masking = gr.Checkbox(
                        label="Enable Court Masking",
                        value=True,
                        info="Mask regions outside the focused court"
                    )
                    
                    enable_tracking = gr.Checkbox(
                        label="Enable Player Tracking",
                        value=True,
                        info="Track players with consistent IDs"
                    )
                    
                    show_boundary = gr.Checkbox(
                        label="Show Court Boundary",
                        value=False,
                        info="Draw court boundary outline"
                    )
                
                gr.Markdown("### 🎛️ Advanced Settings")
                
                with gr.Accordion("Advanced Options", open=False):
                    model_dropdown = gr.Dropdown(
                        choices=YOLO_MODELS,
                        value="yolov8n",
                        label="YOLO Model",
                        info="Larger models are more accurate but slower"
                    )
                    
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Detection Confidence",
                        info="Higher = fewer false positives"
                    )
                    
                    bg_color_input = gr.Textbox(
                        value="0,0,0",
                        label="Background Color (R,G,B)",
                        info="Color for masked regions (e.g., 0,0,0 for black)"
                    )
                
                process_btn = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Processed Video")
                
                video_output = gr.Video(
                    label="Output Video",
                    interactive=False
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
        
        # Examples section
        gr.Markdown("---")
        gr.Markdown(
            """
            ### 📖 Instructions
            
            1. **Upload** a badminton match video (MP4, AVI, MOV, etc.)
            2. **Configure** processing options:
               - Enable/disable court masking
               - Enable/disable player tracking
               - Adjust detection sensitivity
            3. **Click** "Process Video" and wait for processing
            4. **Download** the processed video
            
            ### ⚡ Tips
            
            - Use **yolov8n** for fastest processing, **yolov8x** for best accuracy
            - Lower confidence threshold if players are not being detected
            - Processing time depends on video length and model size
            """
        )
        
        # Connect button to processing function
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                enable_masking,
                enable_tracking,
                show_boundary,
                model_dropdown,
                confidence_slider,
                bg_color_input
            ],
            outputs=[video_output, status_output]
        )
    
    return app


def main():
    """Launch the web UI."""
    print("=" * 50)
    print("🏸 Badminton Court Detection - Web UI")
    print("=" * 50)
    
    app = create_ui()
    
    # Launch with share=False for local use
    # Set share=True to create a public link
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
