#include <cstdlib>
#include <iostream>
#include <cuda_helpers.cuh>
#include <opencv2/opencv.hpp>
#include <common.hpp>
#include "clahe.cuh"

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

template <typename T>
constexpr static inline T sqr(const T &x) {
    return x * x;
}

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

static std::vector<cv::Mat> claheCudaOpencv(const std::vector<cv::Mat> &in, double clipLimit, const cv::Size &gridSize) {
    std::vector<cv::Mat> out(in.size());
    auto clahe = cv::cuda::createCLAHE(clipLimit, gridSize);
    for (int i = 0; i < in.size(); ++i) {
        cv::cuda::GpuMat tmp_in(in[i]), tmp_out;
        clahe->apply(tmp_in, tmp_out);
        tmp_out.download(out[i]);
    }
    return out;
}

static std::vector<cv::Mat> claheCuda(const std::vector<cv::Mat> &in, double clipLimit, const cv::Size &gridSize) {
    std::vector<cv::Mat> out(in.size());
    auto clahe = ::createCLAHE(clipLimit, gridSize);
    for (int i = 0; i < in.size(); ++i) {
        cv::cuda::GpuMat tmp_in(in[i]), tmp_out;
        clahe->apply(tmp_in, tmp_out);
        tmp_out.download(out[i]);
    }
    return out;
}

static void warmupGPU(const cv::Mat &frame) {
    std::vector<cv::Mat> data = { frame, frame, frame, frame };
    claheCudaOpencv(data, 40, cv::Size(8, 8));
}

static cv::Mat concat(const cv::Mat &topLeft, const cv::Mat &topRight, const cv::Mat &bottomLeft, const cv::Mat &bottomRight) {
    cv::Mat top, bottom, output;
    cv::hconcat(topLeft, topRight, top);
    cv::hconcat(bottomLeft, bottomRight, bottom);
    cv::vconcat(top, bottom, output);
    return output;
}

static void drawGrid(cv::Mat &mat, const cv::Size &grid, const cv::Scalar &color=cv::Scalar(0)) {

    int hDist = mat.cols / grid.width;
    int vDist = mat.rows / grid.height;

    for(int i = vDist; i < mat.rows; i += vDist)
        cv::line(mat, cv::Point(0, i), cv::Point(mat.cols, i), color);

    for(int i = hDist; i < mat.cols; i += hDist)
        cv::line(mat, cv::Point(i, 0), cv::Point(i, mat.rows), color);
}

static double mse(const cv::Mat &a, const cv::Mat &b) {
    CV_Assert(a.rows == b.rows && a.cols == b.cols);

    double accum = 0.0;
    for (int i = 0; i < a.rows; ++i) {
        auto a_row = a.ptr<uchar>(i);
        auto b_row = b.ptr<uchar>(i);
        for (int j = 0; j < a.cols; ++j) {
            accum += sqr(a_row[j] - b_row[j]);
        }
    }

    return accum / (a.rows * a.cols);
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
    bool grid = false;
    bool showOutput = false;

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
        } else if (args[i] == "--grid") {
            grid = true;
        } else if (args[i] == "--show-output") {
            showOutput = true;
        } else {
            std::cout << "unrecognized option '" << args[i] << '\'' << std::endl;
        }
    }

    std::cout << "numFrames=" << numFrames << std::endl;
    std::cout << "clipLimit=" << clipLimit << std::endl;
    std::cout << "gridSize=" << gridSize << std::endl;
    std::cout << "grid=" << std::boolalpha << grid << std::endl;

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

    TIMERSTART(warmingGPU);
    warmupGPU(frame);
    TIMERSTOP(warmingGPU);

    std::vector<cv::Mat> m(5);
    m[0] = frame;
    TIMERSTART(equalizeHistogram);
    m[1] = equalizeHistogram(data)[0];
    TIMERSTOP(equalizeHistogram);

    TIMERSTART(claheNative);
    m[2] = claheNative(data, clipLimit, gridSize)[0];
    TIMERSTOP(claheNative);

    TIMERSTART(claheCudaOpencv);
    m[3] = claheCudaOpencv(data, clipLimit, gridSize)[0];
    TIMERSTOP(claheCudaOpencv);

    TIMERSTART(claheCuda);
    m[4] = claheCuda(data, clipLimit, gridSize)[0];
    TIMERSTOP(claheCuda);

    std::cout << std::endl;
    std::cout << "MSE (claheNative/claheCudaOpenCV): " << mse(m[2], m[3]) << std::endl;
    std::cout << "MSE (claheNative/claheCuda)      : " << mse(m[2], m[4]) << std::endl;
    std::cout << "MSE (claheCudaOpenCV/claheCuda)  : " << mse(m[3], m[4]) << std::endl;

    if (showOutput) {
        cv::namedWindow("output", cv::WINDOW_KEEPRATIO);

        for (int i = 0; i < m.size(); ++i) {
            cv::Mat tmp;
            double factor = 384.0 / double(m[i].rows);
            cv::resize(m[i], tmp, cv::Size(), factor, factor);
            m[i] = tmp;
        }

        // draw the grid used for clahe on the image
        if (grid) {
            drawGrid(m[2], gridSize);
            drawGrid(m[4], gridSize);
        }

        cv::Mat output = concat(m[0], m[1], m[2], m[4]);

        cv::imshow("output", output);
        cv::waitKey(0);
    }

    return EXIT_SUCCESS;
}
