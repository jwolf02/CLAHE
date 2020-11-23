#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_helpers.cuh"
#include "common.hpp"

#define DEFAULT_NUM_FRAMES  1
#define DEFAULT_CLIP_LIMIT  40.0
#define DEFAULT_GRID_SIZE   8

#define OUTPUT_IMAGE_HEIGHT 384
#define OUTPUT_WINDOW_NAME  "output"

template <typename T>
constexpr static inline T sqr(const T &x) {
    return x * x;
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
    int gX = DEFAULT_GRID_SIZE, gY = DEFAULT_GRID_SIZE, gZ = DEFAULT_GRID_SIZE;
    bool grid = false;
    bool showOutput = false;
    double scale = 1.0;

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
                gX = gY = gZ = gs;
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
        } else {
            std::cout << "unrecognized option '" << args[i] << '\'' << std::endl;
        }
    }

    std::cout << "numFrames=" << numFrames << std::endl;
    std::cout << "scale=" << scale << std::endl;
    std::cout << "clipLimit=" << clipLimit << std::endl;
    std::cout << "gridSize=(" << gX << ", " << gY << ", " << gZ << ")" << std::endl;
    std::cout << "grid=" << std::boolalpha << grid << std::endl;

    const std::string &filename = args[args.size() - 1];
    cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (frame.empty()) {
        std::cout << "failed to read source" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat target = cv::imread("../out.png", cv::IMREAD_GRAYSCALE);

    if (scale != 1.0) {
      cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    std::cout << "imageSize=[" << frame.rows << " x " << frame.cols << ']' << std::endl;

    auto clahe_cuda = cv::cuda::createCLAHE3D(clipLimit, cv::cuda::Size3i(gX, gY, gZ));

    std::vector<cv::Mat> in(std::max<int>(numFrames, gZ));
    std::vector<cv::Mat> out;
    for (int i = 0; i < in.size(); ++i) {
      in[i] = frame;
    }

    TIMERSTART(CLAHE3DCuda)
    clahe_cuda->apply(in, out);
    TIMERSTOP(CLAHE3DCuda)

    if (showOutput) {
      const double factor = double(OUTPUT_IMAGE_HEIGHT) / double(frame.rows);
      cv::Mat output;
      cv::resize(frame, frame, cv::Size(), factor, factor);
      cv::resize(out[0], out[0], cv::Size(), factor, factor);
      if (!target.empty()) {
        std::cout << rms(out[0], target) << std::endl;
      }
      std::cout << avg(out[0]) << std::endl;
      drawGrid(out[0], cv::Size(gX, gY));
      cv::hconcat(target.empty() ? frame : target, out[0], output);
      cv::namedWindow(OUTPUT_WINDOW_NAME, cv::WINDOW_KEEPRATIO);
      cv::imshow(OUTPUT_WINDOW_NAME, output);
      cv::waitKey(0);
        /*
        for (int i = 0; i < m.size(); ++i) {
            cv::Mat tmp;

            cv::resize(m[i], tmp, cv::Size(), factor, factor);
            m[i] = tmp;
        }

        // draw the grid used for clahe on the image
        if (grid) {
            drawGrid(m[2], gridSize);
            drawGrid(m[4], gridSize);
        }

        cv::Mat output = concat(m[0], m[1], m[2], m[4]);
        */
    }

    return EXIT_SUCCESS;
}
