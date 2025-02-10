# Halftone Vision

Halftone Vision is a CUDA-accelerated tool that applies artistic halftone effects to images and videos. Using both black & white and color modes, this project leverages NVIDIA CUDA and OpenCV to generate halftone effects in real time as well as for offline processing.

## Features

- **CUDA Acceleration:** Uses NVIDIA CUDA to dramatically speed up halftone processing for high-resolution images and videos.
- **Dual Modes:**
  - **Black & White Mode:** Generates a classic halftone effect by varying dot sizes based on brightness.
  - **Color Mode:** Applies halftone effects while preserving color information by averaging colors in each block.
- **Real-Time Processing:** View the halftone effect in real time with interactive adjustments.
- **Interactive Controls:**
  - **Trackbars:** Adjust parameters such as "Dot Size" and "Mode (0:BW, 1:Color)".
  - **Video Navigation:** Seek through video frames using a "Frame" trackbar.
  - **Keyboard Shortcuts:** Toggle pause/play (P), step frame-by-frame (A for backward, D for forward), and save output.
- **Offline Video Processing:** Process and save the entire video offline with your selected parameters (by pressing **S** in video mode). This reprocesses the full video—not just a recording of the display—to produce a high-quality output.
- **Automatic Downscaling:** High-resolution content (e.g., 4K) is automatically downscaled to a maximum resolution of 1920×1080 to ensure compatibility with your display.

## Requirements

- **CUDA Toolkit:** NVIDIA CUDA Toolkit must be installed and configured.
- **OpenCV:** OpenCV (version 4 or later recommended) must be installed.
- **C++ Compiler:** A CUDA-compatible compiler such as `nvcc` is required.

## Compilation

Compile Halftone Vision using NVIDIA’s `nvcc` compiler. For example:

```bash
nvcc -std=c++11 -O2 halftone_vision.cu -o halftone_vision `pkg-config --cflags --libs opencv4`
```

Ensure that your `pkg-config` is properly configured for your OpenCV installation.

## Usage

### Image Mode

To process an image, run:

```bash
./halftone_vision myimage.jpg
```

**Interactive Controls:**
- **Dot Size Trackbar:** Adjusts the size of the halftone dots.
- **Mode Trackbar:** Switch between Black & White (0) and Color (1) modes.

**Keyboard Shortcuts:**
- **S:** Save the current processed image as `halftone_output.png`.
- **ESC:** Exit the application.

### Video Mode

To process a video, run:

```bash
./halftone_vision myvideo.mp4
```

**Interactive Controls:**
- **Frame Trackbar:** Seek through the video.
- **Dot Size & Mode Trackbars:** Adjust the halftone parameters in real time.

**Keyboard Shortcuts:**
- **P:** Toggle pause/play.
- **A:** Step backward one frame (when paused).
- **D:** Step forward one frame (when paused).
- **S:** Process and save the full video offline using the current parameters. The output is saved as `halftone_output.avi`.
- **ESC:** Exit the application.

## How It Works

Halftone Vision divides the input image or video frame into blocks (cells). For each cell:
- In **Black & White Mode**, it computes the average brightness and draws a dot whose radius is proportional to the darkness of the cell.
- In **Color Mode**, it computes the average color and draws a colored dot accordingly.

CUDA kernels accelerate these computations, ensuring high-performance processing even for high-resolution content.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **NVIDIA CUDA:** For providing a powerful platform for GPU computing.
- **OpenCV:** For the robust computer vision framework used for image and video handling.

Happy halftoning!
