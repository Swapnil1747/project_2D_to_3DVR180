# 2D to VR180 Video Converter

A CPU-friendly web application built with Streamlit that converts standard 2D MP4 videos into VR180 stereoscopic videos using depth estimation with the MiDaS model.

## Features

- **Easy Upload**: Upload your 2D MP4 video directly through the web interface.
- **Depth Estimation**: Utilizes MiDaS v2.1 Small model for accurate depth map generation.
- **Stereoscopic Conversion**: Generates left and right eye views for VR180 format.
- **CPU-Friendly**: Optimized for CPU processing, no GPU required.
- **Interactive UI**: Built with Streamlit for a user-friendly experience.
- **Download Output**: Download the converted VR180 video after processing.
- **Adjustable Parameters**: Slider to adjust 3D eye distance for customization.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or higher
- FFmpeg (for video processing)
  - Download from [FFmpeg official website](https://ffmpeg.org/download.html)
  - Add FFmpeg to your system PATH

## Installation

1. Clone or download this repository to your local machine.

2. Navigate to the project directory:
   ```
   cd project_2D_to_3D
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the provided local URL (usually `http://localhost:8501`).

3. Upload a 2D MP4 video using the file uploader.

4. Adjust the 3D eye distance slider if desired (default: 10 pixels).

5. Wait for the processing to complete. The app will:
   - Extract frames from the video
   - Generate depth maps using MiDaS
   - Create stereoscopic frames
   - Reassemble into VR180 video

6. Once processing is done, preview the VR180 video and download it using the provided button.

## How It Works

1. **Frame Extraction**: Uses FFmpeg to extract individual frames from the input video.
2. **Depth Estimation**: Applies MiDaS model to each frame to generate depth maps.
3. **Stereoscopic Generation**: Creates left and right eye views by shifting pixels based on depth.
4. **Video Reassembly**: Combines stereoscopic frames back into a VR180 video using FFmpeg.

## Dependencies

- streamlit
- torch
- torchvision
- opencv-python
- numpy
- ffmpeg-python

## Project Structure

```
twoDto3D/
├── app.py                 # Main Streamlit application
├── .gitignore            # Git ignore file
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── input.mp4             # Sample input video
├── vr180_output.mp4      # Generated output video
├── depth_maps/           # Generated depth maps (PNG files)
├── frames/               # Extracted video frames
├── stereo_frames/        # Generated stereoscopic frames
├── output/               # Output directory
└── myenv/                # Virtual environment (ignored by git)
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MiDaS model by Intel ISL
- Streamlit for the web framework
- FFmpeg for video processing
