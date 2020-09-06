#include <cstdlib>
#include <iostream>
#include <cuda_helpers.cuh>
#include <opencv2/opencv.hpp>

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

static cv::Mat equalizeHistogram(cv::InputArray in) {
    cv::Mat out;
    cv::equalizeHist(in, out);
    return out;
}

static cv::Mat claheNative(cv::InputArray in, double clipLimit=40.0, const cv::Size &gridSize=cv::Size(8, 8)) {
    cv::Mat out;
    auto clahe = cv::createCLAHE(clipLimit, gridSize);
    clahe->apply(in, out);
    return out;
}

static cv::Mat claheCuda(cv::InputArray in, double clipLimit=40.0, const cv::Size &gridSize=cv::Size(8, 8)) {
    cv::Mat out;
    auto clahe = cv::cuda::createCLAHE(clipLimit, gridSize);
    clahe->apply(in, out);
    return out;
}

int main(int argc, const char *argv[]) {
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() == 1) {
        std::cout << "Usage: " << args[0] << "[options...] <source>" << std::endl;
        return EXIT_SUCCESS;
    }

    const std::string &filename = args[1];
    cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (frame.empty()) {
        std::cout << "failed to read source" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat m0, m1, m2;
    TIMERSTART(equalizeHistogram);
    m0 = equalizeHistogram(frame);
    TIMERSTOP(equalizeHistogram);

    TIMERSTART(claheNative);
    m1 = claheNative(frame);
    TIMERSTOP(claheNative);

    TIMERSTART(claheCuda);
    m2 = claheCuda(frame);
    TIMERSTOP(claheCuda);

    cv::namedWindow("equalizeHistogram");
    cv::namedWindow("claheNative");
    cv::namedWindow("claheCuda");

    cv::imshow("equalizeHistogram", m0);
    cv::imshow("claheNative", m1);
    cv::imshow("claheCuda", m2);

    return EXIT_SUCCESS;
}
