#!/usr/bin/env python3
"""
Badminton Court Detection - Fast Web UI

Optimized for speed with frame skipping and simple detection.
"""

import gradio as gr
import tempfile
import os
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def process_video_fast(
    video_file,
    frame_skip: int,
    resize_percent: int,
    enable_tracking: bool,
    model_name: str,
    confidence: float,
    manual_crop: str,
    progress=gr.Progress()
):
    """
    Process video with speed optimizations.
    """
    if video_file is None:
        return None, "❌ Please upload a video file."
    
    try:
        progress(0.05, desc="Initializing...")
        
        from src.fast_pipeline import FastProcessor
        
        # Calculate resize factor
        resize_factor = resize_percent / 100.0
        
        # Parse manual crop if provided
        court_bbox = None
        if manual_crop and manual_crop.strip():
            try:
                parts = [int(x.strip()) for x in manual_crop.split(',')]
                if len(parts) == 4:
                    court_bbox = tuple(parts)  # (x, y, width, height)
                    print(f"Using manual court region: {court_bbox}")
            except:
                pass
        
        progress(0.1, desc="Setting up processor...")
        
        # Create processor with optimizations
        processor = FastProcessor(
            frame_skip=frame_skip,
            resize_factor=resize_factor,
            enable_tracking=enable_tracking,
            mask_color=(0, 0, 0),
            confidence=confidence,
            model=model_name
        )
        
        # Set manual court bbox if provided
        if court_bbox:
            processor._manual_court_bbox = court_bbox
        
        # Create output path
        output_dir = tempfile.mkdtemp()
        output_filename = f"processed_{Path(video_file).stem}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        progress(0.15, desc="Processing video (this may take a moment)...")
        
        # Progress callback
        def update_progress(current, total):
            pct = 0.15 + (current / total) * 0.80
            progress(pct, desc=f"Frame {current}/{total}")
        
        # Process
        success, actual_output = processor.process_video(
            input_path=video_file,
            output_path=output_path,
            progress_callback=update_progress
        )
        
        if success and actual_output and os.path.exists(actual_output):
            file_size = os.path.getsize(actual_output)
            progress(1.0, desc="Complete!")
            return actual_output, f"✅ Processing complete! Output: {os.path.basename(actual_output)} ({file_size/1024/1024:.1f} MB)"
        else:
            return None, "❌ Processing failed. Check logs."
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def create_ui():
    """Create the Gradio web interface."""
    
    with gr.Blocks(title="Badminton Court Detection - Fast Mode") as app:
        
        gr.Markdown(
            """
            # 🏸 Badminton Court Detection - FAST MODE
            
            **Optimized for speed!** Upload a video to detect and isolate the court with player tracking.
            
            ⚡ **Speed Tips:**
            - Higher frame skip = faster (but less smooth)
            - Lower resolution = faster
            - Disable tracking = fastest
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
                
                gr.Markdown("### ⚡ Speed Settings")
                
                frame_skip = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Frame Skip",
                    info="Process every Nth frame. Higher = faster but less smooth."
                )
                
                resize_percent = gr.Slider(
                    minimum=25,
                    maximum=100,
                    value=50,
                    step=25,
                    label="Resolution %",
                    info="Lower = faster. 50% is good balance."
                )
                
                enable_tracking = gr.Checkbox(
                    label="Enable Player Tracking",
                    value=True,
                    info="Disable for fastest processing"
                )
                
                with gr.Accordion("Advanced", open=False):
                    model_dropdown = gr.Dropdown(
                        choices=["yolov8n", "yolov8s", "yolov8m"],
                        value="yolov8n",
                        label="YOLO Model",
                        info="yolov8n = fastest"
                    )
                    
                    confidence_slider = gr.Slider(
                        minimum=0.2,
                        maximum=0.8,
                        value=0.4,
                        step=0.1,
                        label="Detection Confidence"
                    )
                    
                    manual_crop = gr.Textbox(
                        label="Manual Court Region (optional)",
                        placeholder="x, y, width, height (e.g. 200, 50, 1400, 900)",
                        info="Leave empty for auto-detection. Use to manually specify crop region."
                    )
                
                process_btn = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    ---
                    **Estimated Time:**
                    - 1 min video @ 50% res, skip 3: ~30 sec
                    - 1 min video @ 100% res, skip 1: ~3-5 min
                    """
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
                    lines=2
                )
        
        # Connect button
        process_btn.click(
            fn=process_video_fast,
            inputs=[
                video_input,
                frame_skip,
                resize_percent,
                enable_tracking,
                model_dropdown,
                confidence_slider,
                manual_crop
            ],
            outputs=[video_output, status_output]
        )
    
    return app


def main():
    """Launch the web UI."""
    print("=" * 50)
    print("🏸 Badminton Court Detection - FAST MODE")
    print("=" * 50)
    
    app = create_ui()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=None,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
