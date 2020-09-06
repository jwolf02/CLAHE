#include <cstdlib>
#include <iostream>
#include <cuda_helpers.cuh>
#include <opencv2/opencv.hpp>
#include <common.hpp>

#ifdef CUDA_SHARED_MEMORY
#define CUDA_MALLOC(x, len)         x = malloc(len)
#define CUDA_FREE(x)                free((void*) x); x = nullptr;
#define CUDA_H2D(dest, src, len)    memcpy(dest, src, len);
#define CUDA_D2H(dest, src, len)    memcpy(dest, src, len);
#else
#define CUDA_MALLOC(x, len)         cudaMalloc(&x, len); CUERR
#define CUDA_FREE(x)                cudaFree(x); x = nullptr; CUERR
#define CUDA_H2D(dest, src, len)    cudaMemcpy(dest, src, len, H2D);
#define CUDA_D2H(dest, src, len)    cudaMemcpy(dest, src, len, D2H);
#endif

#define DEFAULT_NUM_FRAMES  1
#define DEFAULT_CLIP_LIMIT  40.0
#define DEFAULT_GRID_SIZE   8

static std::vector<cv::Mat> equalizeHistogram(const std::vector<cv::Mat> &in) {
    std::vector<cv::Mat> out(in.size());
    for (int i = 0; i < in.size(); ++i) {
        cv::equalizeHist(in[i], out[i]);
    }
    return out;
}

static std::vector<cv::Mat> claheNative(const std::vector<cv::Mat> &in, double clipLimit, const cv::Size &gridSize) {
    std::vector<cv::Mat> out(in.size());
    auto clahe = cv::createCLAHE(clipLimit, gridSize);
    for (int i = 0; i < in.size(); ++i) {
        clahe->apply(in[i], out[i]);
    }
    return out;
}

static std::vector<cv::Mat> claheCuda(const std::vector<cv::Mat> &in, double clipLimit, const cv::Size &gridSize) {
    std::vector<cv::Mat> out(in.size());
    auto clahe = cv::cuda::createCLAHE(clipLimit, gridSize);
    for (int i = 0; i < in.size(); ++i) {
        cv::cuda::GpuMat tmp_in(in[i]), tmp_out;
        clahe->apply(tmp_in, tmp_out);
        tmp_out.download(out[i]);
    }
    return out;
}

static cv::Mat concat(const cv::Mat &topLeft, const cv::Mat &topRight, const cv::Mat &bottomLeft, const cv::Mat &bottomRight) {
    cv::Mat top, bottom, output;
    cv::hconcat(topLeft, topRight, top);
    cv::hconcat(bottomLeft, bottomRight, bottom);
    cv::vconcat(top, bottom, output);
    return output;
}

int main(int argc, const char *argv[]) {
    const std::vector<std::string> args(argv, argv + argc);
    if (args.size() == 1) {
        std::cout << "Usage: " << args[0] << " [options...] <source>" << std::endl;
        return EXIT_SUCCESS;
    }

    unsigned numFrames = DEFAULT_NUM_FRAMES;
    double clipLimit = DEFAULT_CLIP_LIMIT;
    cv::Size gridSize = cv::Size(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE);

    for (int i = 1; i < args.size() - 1; ++i) {
        if (string::starts_with(args[i], "--clip-limit=")) {
            auto tokens = string::split(args[i], "=");
            if (tokens.size() == 2) {
                clipLimit = string::to<double>(tokens[1]);
            }
        } else if (string::starts_with(args[i], "--grid-size=")) {
            auto tokens = string::split(args[i], "=");
            if (tokens.size() == 2) {
                auto gs = string::to<unsigned>(tokens[1]);
                gridSize = cv::Size(gs, gs);
            }
        } else if (string::starts_with(args[i], "--num-frames=")) {
            auto tokens = string::split(args[i], "=");
            if (tokens.size() == 2) {
                numFrames = string::to<unsigned>(tokens[1]);
            }
        } else {
            std::cout << "unrecognized option '" << args[i] << '\'' << std::endl;
        }
    }

    std::cout << "numFrames=" << numFrames << std::endl;
    std::cout << "clipLimit=" << clipLimit << std::endl;
    std::cout << "gridSize=" << gridSize << std::endl;

    const std::string &filename = args[args.size() - 1];
    cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (frame.empty()) {
        std::cout << "failed to read source" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "imageSize=[" << frame.rows << " x " << frame.cols << ']' << std::endl;

    std::vector<cv::Mat> data(numFrames);
    for (int i = 0; i < data.size(); ++i) {
        frame.copyTo(data[i]);
    }

    std::cout << std::endl;

    std::vector<cv::Mat> m(3);
    TIMERSTART(equalizeHistogram);
    m[0] = equalizeHistogram(data)[0];
    TIMERSTOP(equalizeHistogram);

    TIMERSTART(claheNative);
    m[1] = claheNative(data, clipLimit, gridSize)[0];
    TIMERSTOP(claheNative);

    TIMERSTART(claheCuda);
    m[2] = claheCuda(data, clipLimit, gridSize)[0];
    TIMERSTOP(claheCuda);

    cv::namedWindow("output", cv::WINDOW_AUTOSIZE);

    cv::Mat output = concat(frame, m[0], m[1], m[2]);

    cv::imshow("output", output);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
