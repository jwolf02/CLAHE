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



template <typename T>
constexpr static inline T sqr(const T &x) {
    return x * x;
}

static cv::Mat equalizeHistogram(const cv::Mat &in, int reps=1) {
    cv::Mat out;
    for (int i = 0; i < reps; ++i) {
        cv::equalizeHist(in, out);
    }
    return out;
}

static cv::Mat claheNative(const cv::Mat &in, double clipLimit, const cv::Size &gridSize, int reps=1) {
    cv::Mat out;
    auto clahe = cv::createCLAHE(clipLimit, gridSize);
    for (int i = 0; i < reps; ++i) {
        clahe->apply(in, out);
    }
    return out;
}

static cv::Mat claheCudaOpencv(const cv::Mat &in, double clipLimit, const cv::Size &gridSize, int reps=1) {
    cv::Mat out;
    auto clahe = cv::cuda::createCLAHE(clipLimit, gridSize);
    cv::cuda::GpuMat tmp_in(in), tmp_out;
    TIMERSTART(claheCudaOpencv);
    for (int i = 0; i < reps; ++i) {
        clahe->apply(tmp_in, tmp_out);
    }
    TIMERSTOP(claheCudaOpencv);
    tmp_out.download(out);
    return out;
}

static cv::Mat claheCuda(const cv::Mat &in, double clipLimit, const cv::Size &gridSize, int reps=1) {
    cv::Mat out;
    auto clahe = cv::cuda::createCLAHE(clipLimit, gridSize);
    cv::cuda::GpuMat tmp_in(in), tmp_out;
    TIMERSTART(claheCuda);
    for (int i = 0; i < reps; ++i) {
        clahe->apply(tmp_in, tmp_out);
    }
    TIMERSTOP(claheCuda);
    tmp_out.download(out);
    return out;
}

static void warmupGPU(const cv::Mat &frame) {
    claheCudaOpencv(frame, 40, cv::Size(8, 8), 4);
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
            accum += sqr(double(a_row[j]) - double(b_row[j]));
        }
    }

    return accum / (a.rows * a.cols);
}

static double rms(const cv::Mat &a, const cv::Mat &b) {
    return std::sqrt(mse(a, b));
}

static double avg(const cv::Mat &a) {
  double accum = 0;
  for (int i = 0; i < a.rows; ++i) {
    auto ptr = a.ptr(i);
    for (int j = 0; j < a.cols; ++j) {
      accum += (double) ptr[j];
    }
  }
  return accum / double(a.rows * a.cols);
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
    double scale = 1.0;
    bool saveOutput = false;

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
        } else if (string::starts_with(args[i], "--scale=")) {
            auto tokens = string::split(args[i], "=");
            if (tokens.size() == 2) {
                scale = string::to<double>(tokens[1]);
            }
        } else if (args[i] == "--grid") {
            grid = true;
        } else if (args[i] == "--show-output") {
            showOutput = true;
        } else if (args[i] == "--save-output") {
            saveOutput = true;
        } else {
            std::cout << "unrecognized option '" << args[i] << '\'' << std::endl;
        }
    }

    std::cout << "numFrames=" << numFrames << std::endl;
    std::cout << "scale=" << scale << std::endl;
    std::cout << "clipLimit=" << clipLimit << std::endl;
    std::cout << "gridSize=" << gridSize << std::endl;
    std::cout << "grid=" << std::boolalpha << grid << std::endl;

    const std::string &filename = args[args.size() - 1];
    cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (frame.empty()) {
        std::cout << "failed to read source" << std::endl;
        return EXIT_FAILURE;
    }

    if (scale != 1.0) {
        cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    std::cout << "imageSize=[" << frame.rows << " x " << frame.cols << ']' << std::endl;

    TIMERSTART(warmingGPU);
    warmupGPU(frame);
    TIMERSTOP(warmingGPU);

    std::vector<cv::Mat> m(5);
    m[0] = frame;
    TIMERSTART(equalizeHistogram);
    m[1] = equalizeHistogram(frame, numFrames);
    TIMERSTOP(equalizeHistogram);

    TIMERSTART(claheNative);
    m[2] = claheNative(frame, clipLimit, gridSize, numFrames);
    TIMERSTOP(claheNative);

    m[3] = claheCudaOpencv(frame, clipLimit, gridSize, numFrames);

    m[4] = claheCuda(frame, clipLimit, gridSize, numFrames);

    std::cout << "RMS(src, claheCuda)            =" << rms(m[0], m[4]) << std::endl;
    std::cout << "RMS(histEq, claheCuda)         =" << rms(m[1], m[4]) << std::endl;
    std::cout << "RMS(native, claheCudaOpencv)   =" << rms(m[2], m[3]) << std::endl;
    std::cout << "RMS(claheCudaOpencv, claheCuda)=" << rms(m[3], m[4]) << std::endl;

    std::cout << "AVG(claheCudaOpencv)           =" << avg(m[3]) << std::endl;
    std::cout << "AVG(claheCuda)                 =" << avg(m[4]) << std::endl;

    if (showOutput) {

    }

    return EXIT_SUCCESS;
}
