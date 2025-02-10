// File: halftone_vision.cu
// Compile with: nvcc -std=c++11 -O2 dot_video.cu -o dot_video `pkg-config --cflags --libs opencv4`

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <algorithm>

// --------------------------------------------------------------------------
// Utility: CUDA error checking macro
// --------------------------------------------------------------------------
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --------------------------------------------------------------------------
// Structure to hold average color for a block (used in Color mode)
// --------------------------------------------------------------------------
struct ColorAvg {
    float r;
    float g;
    float b;
    float luminance;  // computed as 0.299*r + 0.587*g + 0.114*b
};

// --------------------------------------------------------------------------
// CUDA Kernels for Black & White Halftone Effect
// --------------------------------------------------------------------------

// Kernel 1: Compute average brightness for each cell in a grayscale image.
__global__ void computeBlockAveragesBW(const unsigned char* gray, float* blockAverages,
                                         int width, int height, int cellSize)
{
    int bx = blockIdx.x; // cell (block) x index
    int by = blockIdx.y; // cell (block) y index
    int numBlocksX = gridDim.x; // total number of cells in x direction
    int blockIndex = by * numBlocksX + bx;
    int startX = bx * cellSize;
    int startY = by * cellSize;
    float sum = 0.0f;
    int count = 0;
    for (int y = startY; y < startY + cellSize && y < height; y++) {
        for (int x = startX; x < startX + cellSize && x < width; x++) {
            sum += gray[y * width + x];
            count++;
        }
    }
    blockAverages[blockIndex] = (count > 0) ? (sum / count) : 255.0f;
}

// Kernel 2: For each pixel in the output, draw a black dot on white if it lies inside
// the computed circle (whose radius is proportional to (1 - brightness)).
__global__ void applyHalftoneBW(const float* blockAverages, unsigned char* output,
                                int width, int height, int cellSize, int dotSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int numBlocksX = (width + cellSize - 1) / cellSize;
    int bx = x / cellSize;
    int by = y / cellSize;
    int blockIndex = by * numBlocksX + bx;
    float avg = blockAverages[blockIndex];
    // Compute dot radius: darker cells yield a larger dot.
    float r = dotSize * (1.0f - avg / 255.0f);
    float centerX = bx * cellSize + cellSize / 2.0f;
    float centerY = by * cellSize + cellSize / 2.0f;
    float dx = x - centerX;
    float dy = y - centerY;
    float dist = sqrtf(dx * dx + dy * dy);
    unsigned char color = (dist <= r) ? 0 : 255; // black dot vs white background
    int idx = (y * width + x) * 3;
    output[idx    ] = color;
    output[idx + 1] = color;
    output[idx + 2] = color;
}

// --------------------------------------------------------------------------
// CUDA Kernels for Color Halftone Effect
// --------------------------------------------------------------------------

// Kernel 1: Compute the average color (B, G, R) for each cell.
__global__ void computeBlockAveragesColor(const unsigned char* input, ColorAvg* blockAvgs,
                                            int width, int height, int cellSize)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int numBlocksX = gridDim.x;
    int blockIndex = by * numBlocksX + bx;
    int startX = bx * cellSize;
    int startY = by * cellSize;
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    int count = 0;
    for (int y = startY; y < startY + cellSize && y < height; y++) {
        for (int x = startX; x < startX + cellSize && x < width; x++) {
            int idx = (y * width + x) * 3;
            // OpenCV images are in BGR order.
            sumB += input[idx];
            sumG += input[idx + 1];
            sumR += input[idx + 2];
            count++;
        }
    }
    ColorAvg avg;
    if (count > 0) {
        avg.r = sumR / count;
        avg.g = sumG / count;
        avg.b = sumB / count;
    } else {
        avg.r = avg.g = avg.b = 255.0f;
    }
    // Compute luminance from the average color.
    avg.luminance = 0.299f * avg.r + 0.587f * avg.g + 0.114f * avg.b;
    blockAvgs[blockIndex] = avg;
}

// Kernel 2: For each pixel, if it falls inside the computed circle for its cell,
// set the pixel to the average color; otherwise, leave the original pixel.
__global__ void applyHalftoneColor(const ColorAvg* blockAvgs, const unsigned char* input,
                                   unsigned char* output, int width, int height,
                                   int cellSize, int dotSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int numBlocksX = (width + cellSize - 1) / cellSize;
    int bx = x / cellSize;
    int by = y / cellSize;
    int blockIndex = by * numBlocksX + bx;
    ColorAvg avg = blockAvgs[blockIndex];
    float r = dotSize * (1.0f - avg.luminance / 255.0f);
    float centerX = bx * cellSize + cellSize / 2.0f;
    float centerY = by * cellSize + cellSize / 2.0f;
    float dx = x - centerX;
    float dy = y - centerY;
    float dist = sqrtf(dx * dx + dy * dy);
    int idx = (y * width + x) * 3;
    if (dist <= r) {
        // Inside dot: output the average color (BGR order)
        output[idx    ] = (unsigned char)min(max(int(avg.b + 0.5f), 0), 255);
        output[idx + 1] = (unsigned char)min(max(int(avg.g + 0.5f), 0), 255);
        output[idx + 2] = (unsigned char)min(max(int(avg.r + 0.5f), 0), 255);
    } else {
        // Outside dot: keep the original pixel.
        output[idx    ] = input[idx    ];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
    }
}

// --------------------------------------------------------------------------
// Helper Function: Resize image if it exceeds a maximum resolution (1920x1080)
// --------------------------------------------------------------------------
cv::Mat resizeIfNeeded(const cv::Mat &input, int maxWidth = 1920, int maxHeight = 1080) {
    cv::Mat resized = input;
    if (input.cols > maxWidth || input.rows > maxHeight) {
        double scaleX = (double)maxWidth / input.cols;
        double scaleY = (double)maxHeight / input.rows;
        double scale = std::min(scaleX, scaleY);
        cv::resize(input, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    return resized;
}

// --------------------------------------------------------------------------
// Host Function: processImage
// Transfers the input image (or video frame) to the GPU, launches the appropriate kernels,
// and returns the processed output as a cv::Mat.
// Parameters:
//    - input: the source image (BGR; if mode==0, converted to grayscale)
//    - dotSize: controls the “dot” size and thus the cell size (cellSize = dotSize*2)
//    - mode: 0 = Black & White halftone, 1 = Color halftone
// --------------------------------------------------------------------------
cv::Mat processImage(const cv::Mat &input, int dotSize, int mode)
{
    // Enforce a minimum dotSize.
    dotSize = (dotSize < 2) ? 2 : dotSize;
    int cellSize = dotSize * 2;
    int width = input.cols;
    int height = input.rows;
    cv::Mat output(height, width, CV_8UC3);

    if (mode == 0) {
        // --- BLACK & WHITE MODE ---
        cv::Mat gray;
        if (input.channels() == 3)
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else
            gray = input;

        size_t graySize = width * height * sizeof(unsigned char);
        size_t outputSize = width * height * 3 * sizeof(unsigned char);

        unsigned char *d_gray = nullptr, *d_output = nullptr;
        float *d_blockAverages = nullptr;
        CUDA_CHECK(cudaMalloc(&d_gray, graySize));
        CUDA_CHECK(cudaMalloc(&d_output, outputSize));
        int numBlocksX = (width + cellSize - 1) / cellSize;
        int numBlocksY = (height + cellSize - 1) / cellSize;
        int numBlocks = numBlocksX * numBlocksY;
        CUDA_CHECK(cudaMalloc(&d_blockAverages, numBlocks * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_gray, gray.data, graySize, cudaMemcpyHostToDevice));

        // Launch kernel to compute average brightness per cell.
        dim3 gridDim(numBlocksX, numBlocksY);
        computeBlockAveragesBW<<<gridDim, 1>>>(d_gray, d_blockAverages, width, height, cellSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Launch kernel to generate the halftone image.
        dim3 blockDim(16, 16);
        dim3 gridDim2((width + blockDim.x - 1) / blockDim.x,
                      (height + blockDim.y - 1) / blockDim.y);
        applyHalftoneBW<<<gridDim2, blockDim>>>(d_blockAverages, d_output, width, height, cellSize, dotSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data, d_output, outputSize, cudaMemcpyDeviceToHost));

        cudaFree(d_gray);
        cudaFree(d_output);
        cudaFree(d_blockAverages);
    }
    else if (mode == 1) {
        // --- COLOR MODE ---
        size_t inputSize = width * height * 3 * sizeof(unsigned char);
        size_t outputSize = inputSize;

        unsigned char *d_input = nullptr, *d_output = nullptr;
        ColorAvg *d_blockAvgs = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, inputSize));
        CUDA_CHECK(cudaMalloc(&d_output, outputSize));
        int numBlocksX = (width + cellSize - 1) / cellSize;
        int numBlocksY = (height + cellSize - 1) / cellSize;
        int numBlocks = numBlocksX * numBlocksY;
        CUDA_CHECK(cudaMalloc(&d_blockAvgs, numBlocks * sizeof(ColorAvg)));

        CUDA_CHECK(cudaMemcpy(d_input, input.data, inputSize, cudaMemcpyHostToDevice));

        // Launch kernel to compute average color per cell.
        dim3 gridDim(numBlocksX, numBlocksY);
        computeBlockAveragesColor<<<gridDim, 1>>>(d_input, d_blockAvgs, width, height, cellSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Launch kernel to generate the color halftone image.
        dim3 blockDim(16, 16);
        dim3 gridDim2((width + blockDim.x - 1) / blockDim.x,
                      (height + blockDim.y - 1) / blockDim.y);
        applyHalftoneColor<<<gridDim2, blockDim>>>(d_blockAvgs, d_input, d_output, width, height, cellSize, dotSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data, d_output, outputSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_blockAvgs);
    }
    return output;
}

// --------------------------------------------------------------------------
// Offline Video Saving Function
// Processes the full video from start to end using the current parameters
// and saves the output as "halftone_output.avi" (after downscaling if needed).
// --------------------------------------------------------------------------
void saveFullVideo(const std::string &videoFilename, int dotSize, int mode) {
    cv::VideoCapture capVideo(videoFilename);
    if (!capVideo.isOpened()) {
        std::cerr << "Error opening video file for offline saving: " << videoFilename << std::endl;
        return;
    }

    // Read one frame to determine output dimensions (and perform downscaling if needed).
    cv::Mat sampleFrame;
    capVideo >> sampleFrame;
    if (sampleFrame.empty()) {
        std::cerr << "Error: video file is empty." << std::endl;
        return;
    }
    sampleFrame = resizeIfNeeded(sampleFrame, 1920*sqrt(2), 1080*sqrt(2));
    int outWidth = sampleFrame.cols;
    int outHeight = sampleFrame.rows;

    // Rewind video to beginning.
    capVideo.set(cv::CAP_PROP_POS_FRAMES, 0);

    int codec = cv::VideoWriter::fourcc('M','J','P','G');
    double fps = capVideo.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;
    cv::VideoWriter writer("halftone_output.avi", codec, fps, cv::Size(outWidth, outHeight), true);
    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video for write." << std::endl;
        return;
    }

    cv::Mat frame, output;
    std::cout << "Processing full video offline..." << std::endl;
    while (true) {
        capVideo >> frame;
        if (frame.empty())
            break;
        frame = resizeIfNeeded(frame, 1920, 1080);
        output = processImage(frame, dotSize, mode);
        writer.write(output);
    }
    writer.release();
    capVideo.release();
    std::cout << "Full video saved as halftone_output.avi" << std::endl;
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
// Usage:
//    chatgpt_ad_maker <input file (image or video)>
//
// For an image, a window with interactive trackbars ("Dot Size" and "Mode")
// is displayed. Press 'S' to save the processed image.
// 
// For a video, the effect is applied frame-by-frame. In video mode:
//    - A "Frame" trackbar lets you seek through the video.
//    - Press 'P' to toggle pause/play.
//    - When paused, press 'A' or 'D' to step backward or forward one frame.
//    - Press 'S' to process and save the full video (offline) using the current parameters.
//    - ESC exits.
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: dot video maker <input file (image or video)>" << std::endl;
        return 0;
    }
    
    std::string filename = argv[1];
    cv::VideoCapture cap(filename);
    bool isVideo = cap.isOpened();

    cv::namedWindow("Halftone", cv::WINDOW_NORMAL);

    // Create trackbars without value pointers to avoid deprecation warnings.
    cv::createTrackbar("Dot Size", "Halftone", nullptr, 50, nullptr);
    cv::setTrackbarPos("Dot Size", "Halftone", 10);
    cv::createTrackbar("Mode (0:BW, 1:Color)", "Halftone", nullptr, 1, nullptr);
    cv::setTrackbarPos("Mode (0:BW, 1:Color)", "Halftone", 0);

    if (!isVideo) {
        // ---- Process a still image ----
        cv::Mat image = cv::imread(filename);
        if (image.empty()) {
            std::cerr << "Error loading image: " << filename << std::endl;
            return 1;
        }
        image = resizeIfNeeded(image, 1920, 1080);

        while (true) {
            int dotSize = cv::getTrackbarPos("Dot Size", "Halftone");
            int mode = cv::getTrackbarPos("Mode (0:BW, 1:Color)", "Halftone");
            cv::Mat output = processImage(image, dotSize, mode);
            cv::imshow("Halftone", output);
            int key = cv::waitKey(30);
            if (key == 27) break;  // ESC to exit
            // In image mode, pressing S saves the current processed image.
            if (key == 's' || key == 'S') {
                cv::imwrite("halftone_output.png", output);
                std::cout << "Saved halftone_output.png" << std::endl;
            }
        }
    }
    else {
        // ---- Process a video file interactively ----
        int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        cv::createTrackbar("Frame", "Halftone", nullptr, totalFrames - 1, nullptr);
        cv::setTrackbarPos("Frame", "Halftone", 0);

        bool paused = false;
        int currentFrame = 0;
        cv::Mat lastFrame;

        while (true) {
            int dotSize = cv::getTrackbarPos("Dot Size", "Halftone");
            int mode = cv::getTrackbarPos("Mode (0:BW, 1:Color)", "Halftone");
            int trackFrame = cv::getTrackbarPos("Frame", "Halftone");

            cv::Mat frame;
            if (!paused) {
                cap >> frame;
                if (frame.empty()) {
                    // End reached: stay paused for frame review.
                    paused = true;
                }
            }
            else {
                // If paused and the "Frame" trackbar is moved, jump to that frame.
                if (trackFrame != currentFrame) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, trackFrame);
                    cap >> frame;
                }
            }
            if (!frame.empty()) {
                lastFrame = frame.clone();
                currentFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
                cv::setTrackbarPos("Frame", "Halftone", currentFrame);
            } else {
                frame = lastFrame.clone();
            }

            frame = resizeIfNeeded(frame, 1920, 1080);
            cv::Mat output = processImage(frame, dotSize, mode);
            cv::imshow("Halftone", output);

            int key = cv::waitKey(30);
            if (key == 27) break;  // ESC to exit

            // Toggle pause/play with 'P'
            if (key == 'p' || key == 'P') {
                paused = !paused;
                cv::setTrackbarPos("Frame", "Halftone", currentFrame);
            }
            // When paused, allow stepping backward with 'A'
            if (key == 'a' || key == 'A') {
                if (paused) {
                    int newFrame = std::max(currentFrame - 1, 0);
                    cap.set(cv::CAP_PROP_POS_FRAMES, newFrame);
                    cv::setTrackbarPos("Frame", "Halftone", newFrame);
                    currentFrame = newFrame;
                    cap >> frame;
                    if (!frame.empty())
                        lastFrame = frame.clone();
                }
            }
            // When paused, allow stepping forward with 'D'
            if (key == 'd' || key == 'D') {
                if (paused) {
                    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
                    int newFrame = std::min(currentFrame + 1, total - 1);
                    cap.set(cv::CAP_PROP_POS_FRAMES, newFrame);
                    cv::setTrackbarPos("Frame", "Halftone", newFrame);
                    currentFrame = newFrame;
                    cap >> frame;
                    if (!frame.empty())
                        lastFrame = frame.clone();
                }
            }
            // In video mode, pressing S processes and saves the full video offline.
            if (key == 's' || key == 'S') {
                std::cout << "Saving full video with parameters: Dot Size = " << dotSize
                          << ", Mode = " << mode << std::endl;
                saveFullVideo(filename, dotSize, mode);
                std::cout << "Full video saved as halftone_output.avi" << std::endl;
            }
        }
        cap.release();
    }
    
    cv::destroyAllWindows();
    return 0;
}
