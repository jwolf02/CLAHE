#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "nifti.hpp"
#include <cuda_helpers.cuh>
#include <dirent.h>
#include <common.hpp>

#define OUTPUT_IMAGE_HEIGHT 384
#define OUTPUT_WINDOW_NAME  "output"

static std::vector<std::string> getAllFiles(const std::string &dirname) {
  DIR *dir;
  struct dirent *ent;
  std::vector<std::string> fnames;
  if ((dir = opendir(dirname.c_str())) != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      fnames.emplace_back(ent->d_name);
    }
    closedir (dir);
  } else {
    throw std::runtime_error("failed to open directory");
  }
  return fnames;
}

static void drawGrid(cv::Mat &mat, const cv::Size &grid, const cv::Scalar &color=cv::Scalar(0)) {
  int hDist = mat.cols / grid.width;
  int vDist = mat.rows / grid.height;

  for(int i = vDist; i < mat.rows; i += vDist)
    cv::line(mat, cv::Point(0, i), cv::Point(mat.cols, i), color);

  for(int i = hDist; i < mat.cols; i += hDist)
    cv::line(mat, cv::Point(i, 0), cv::Point(i, mat.rows), color);
}

static cv::Mat concat(const cv::Mat &topLeft, const cv::Mat &topRight, const cv::Mat &bottomLeft, const cv::Mat &bottomRight) {
  cv::Mat top, bottom, output;
  cv::hconcat(topLeft, topRight, top);
  cv::hconcat(bottomLeft, bottomRight, bottom);
  cv::vconcat(top, bottom, output);
  return output;
}

int main(int argc, char *argv[]) {
  const std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    std::cout << "Usage: " << args[0] << " <dirname> [--clipLimit=X] [--numTiles=X] [--equalizeHist] [--CLAHE] [--CLAHE3D]" << std::endl;
    return 0;
  }
  std::string dirname = args[1];

  int numImages = 1024;
  double clipLimit = 40.0;
  int numTiles = 8;
  bool equalizeHist = false;
  bool useClahe = false;
  bool useClahe3D = false;
  bool texMem = false;
  bool gpu = false;
  for (size_t i = 2; i < args.size(); ++i) {
    const auto &arg = args[i];
    if (string::starts_with(arg, "--clipLimit=")) {
      auto tokens = string::split(arg, "=");
      clipLimit = tokens.size() >= 2 ? strtod(tokens[1].c_str(), nullptr) : clipLimit;
    } else if (string::starts_with(arg, "--numTiles=")) {
      auto tokens = string::split(arg, "=");
      numTiles = tokens.size() >= 2 ? strtol(tokens[1].c_str(), nullptr, 10) : numTiles;
    } else if (arg == "--HE") {
      equalizeHist = true;
    } else if (arg == "--CLAHE") {
      useClahe = true;
    } else if (arg == "--CLAHE3D") {
      useClahe3D = true;
    } else if (arg == "--texMem") {
      texMem = true;
    } else if (arg == "--gpu") {
      gpu = true;
    } else if (string::starts_with(arg, "--numImages=")) {
      auto tokens = string::split(arg, "=");
      numImages = tokens.size() >= 2 ? strtol(tokens[1].c_str(), nullptr, 10) : numImages;
    } else {
      std::cout << "unrecognized option '" << arg << "'" << std::endl;
    }
  }

  /*
  std::vector<std::string> files = getAllFiles(dirname);
  std::vector<std::string> nifti_files;
  std::copy_if(files.begin(), files.end(), std::back_inserter(nifti_files),
               [](const std::string &fname) { return string::ends_with(fname, ".nii"); });
  std::sort(nifti_files.begin(), nifti_files.end());

  std::vector<cv::Mat> frames;

  dirname += '/';

  TIMERSTART(readNIFTI)
  for (const auto &filename : nifti_files) {
    auto f = nifti::read(dirname + filename);
    if (f.empty()) {
      std::cout << "warning: " << filename << " does not contain any data" << std::endl;
    } else {
      frames.reserve(frames.size() + f.size());
      for (size_t i = 0; i < f.size(); ++i) {
        auto &mat = f[i];
        cv::normalize(mat, mat, 0, 255, cv::NORM_MINMAX);
        if (mat.type() != CV_8UC1) {
          mat.convertTo(mat, CV_8UC1);
        }
        frames.push_back(mat);
      }
    }
  }
  TIMERSTOP(readNIFTI)

  int i = 0;
  for (auto &mat : frames) {
    cv::resize(mat, mat, cv::Size(4096, 3072));
    auto fname = std::to_string(i++) + ".png";
    std::cout << "writing " << fname << "..." << std::endl;
    cv::imwrite(fname, mat);
  }*/

  cv::Mat frame;
  frame = cv::imread(dirname, cv::IMREAD_GRAYSCALE);
  if (frame.empty()) {
    std::cout << "failed to read frame" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat out;
  cv::equalizeHist(frame, out);
  cv::imwrite("./HE.png", out);

  auto clahe = cv::createCLAHE(clipLimit, cv::Size(numTiles, numTiles));
  clahe->apply(frame, out);
  cv::imwrite("./CLAHE.png", out);

  return 0;

  if (equalizeHist && !gpu) {
    for (int i = 0; i < 5; ++i) {
      cv::Mat tmp;
      int s = 1 << (10 + i);
      cv::resize(frame, tmp, cv::Size(s, s));
      std::cout << s << "x" << s << std::endl;
      for (int j = 0; j < 5; ++j) {
        TIMERSTART(HE);
        cv::equalizeHist(tmp, tmp);
        TIMERSTOP(HE);
      }
    }
  }

  if (equalizeHist && gpu) {
    for (int i = 0; i < 5; ++i) {
      cv::Mat tmp;
      cv::cuda::GpuMat tmpGpu;
      int s = 1 << (10 + i);
      cv::resize(frame, tmp, cv::Size(s, s));
      std::cout << s << "x" << s << std::endl;
      tmpGpu.upload(tmp);
      for (int j = 0; j < 5; ++j) {
        TIMERSTART(HEgpu);
	cv::cuda::equalizeHist(tmpGpu, tmpGpu);
	TIMERSTOP(HEgpu);
      }
    }
  }

  if (useClahe && !gpu) {
    for (int i = 0; i < 5; ++i) {
      cv::Mat tmp;
      auto clahe = cv::createCLAHE(clipLimit, cv::Size(numTiles, numTiles));
      int s = 1 << (10 + i);
      cv::resize(frame, tmp, cv::Size(s, s));
      std::cout << s << "x" << s << std::endl;
      for (int j = 0; j < 5; ++j) {
        TIMERSTART(CLAHE2Dcpu);
	clahe->apply(tmp, tmp);
	TIMERSTOP(CLAHE2Dcpu);
      }
    }
  }

  if (useClahe && gpu) {
    for (int i = 0; i < 5; ++i) {
      cv::Mat tmp;
      cv::cuda::GpuMat tmpGpu;
      auto clahe = cv::cuda::createCLAHE(clipLimit, cv::Size(numTiles, numTiles));
      int s = (1 << (10 + i)) + 3;
      cv::resize(frame, tmp, cv::Size(s, s));
      std::cout << s << "x" << s << std::endl;
      tmpGpu.upload(tmp);
      for (int j = 0; j < 5; ++j) {
        TIMERSTART(CLAHE2Dgpu);
	clahe->apply(tmpGpu, tmpGpu);
	TIMERSTOP(CLAHE2Dgpu);
      }
    }
  }

  return EXIT_SUCCESS;
}
