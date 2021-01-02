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

  std::vector<cv::Mat> frames;

  dirname += '/';

  TIMERSTART(readImages);
  for (int i = 0; i < numImages; ++i) {
    cv::Mat frame = cv::imread(dirname + std::to_string(i) + ".png", cv::IMREAD_GRAYSCALE );
    if (frame.empty()) {
      break;
    }
    //cv::resize(frame, frame, cv::Size(frame.cols + 1, frame.rows));
    frames.push_back(frame);
  }
  TIMERSTOP(readImages);

  std::cout << "loaded " << frames.size() << " frames" << std::endl;

  CV_Assert(!frames.empty());

  size_t bytes = 0;
  for (auto &mat : frames) {
    cv::resize(mat, mat, cv::Size(frames[0].cols, frames[0].rows));
    bytes += mat.rows * mat.cols;
  }
  std::cout << bytes / 1000 << " KBytes" << std::endl;

  std::vector<cv::Mat> equalizeHistOut;
  std::vector<cv::Mat> clahe2DOut;
  std::vector<cv::Mat> clahe3DOut;

  if (equalizeHist) {
    std::vector<cv::Mat> out(frames.size());
    if (gpu) {
      std::vector<cv::cuda::GpuMat> gpuFrames(frames.size());
      TIMERSTART(upload);
      for (size_t i = 0; i < frames.size(); ++i) {
        gpuFrames[i].upload(frames[i]);
      }
      TIMERSTOP(upload);

      // warmup
      cv::cuda::equalizeHist(gpuFrames[0], gpuFrames[0]);

      TIMERSTART(HEgpu);
      for (size_t i = 0; i < frames.size(); ++i) {
        cv::cuda::equalizeHist(gpuFrames[i], gpuFrames[i]);
      }
      TIMERSTOP(HEgpu);

      for (size_t i = 0; i < frames.size(); ++i) {
        gpuFrames[i].download(out[i]);
      }
    } else {
      TIMERSTART(equalizeHist)
      for (size_t i = 0; i < frames.size(); ++i) {
        cv::equalizeHist(frames[i], out[i]);
      }
      TIMERSTOP(equalizeHist)
    } 
    equalizeHistOut = std::move(out);
  }

  if (useClahe) {
    std::vector<cv::Mat> out(frames.size());
    if (gpu) {
      std::vector<cv::cuda::GpuMat> gpuFrames(frames.size());
      for (size_t i = 0; i < frames.size(); ++i) {
        gpuFrames[i].upload(frames[i]);
      }

      auto clahe = cv::cuda::createCLAHE(clipLimit, cv::Size(numTiles, numTiles));
      
      // warmup
      clahe->apply(gpuFrames[0], gpuFrames[0]);
      
      TIMERSTART(CLAHE2D)
      for (size_t i = 0; i < gpuFrames.size(); ++i) {
        clahe->apply(gpuFrames[i], gpuFrames[i]);
      }
      TIMERSTOP(CLAHE2D)

      for (size_t i = 0; i < frames.size(); ++i) {
        gpuFrames[i].download(out[i]);
      }
    } else {
      auto clahe = cv::createCLAHE(clipLimit, cv::Size(numTiles, numTiles));
      TIMERSTART(CLAHE2D);
      for (size_t i = 0; i < frames.size(); ++i) {
	clahe->apply(frames[i], out[i]);
      }
      TIMERSTOP(CLAHE2D);
    }
    clahe2DOut = std::move(out);
  }

  /*
  if (useClahe3D) {
    std::vector<cv::Mat> out;
    auto clahe = cv::cuda::createCLAHE3D(clipLimit, cv::cuda::Size3i(numTiles, numTiles, numTiles));
    uint8_t *dev_ptr = nullptr;
    uint8_t *dev_out = nullptr;
    size_t fsize = frames[0].rows * frames[0].cols;
    cudaMalloc(&dev_ptr, frames.size() * fsize);
    cudaMalloc(&dev_out, frames.size() * fsize);
    for (size_t i = 0; i < frames.size(); ++i) {
      cudaMemcpy(dev_ptr + i * fsize, frames[0].data, fsize, H2D); 
    }
    TIMERSTART(CLAHE3D);
    clahe->apply(cv::cuda::DevPtr<uchar>(dev_ptr), cv::cuda::DevPtr<uchar>(dev_out), frames[0].rows, frames[0].cols, (int) frames.size(), cv::cuda::Stream::Null());
    TIMERSTOP(CLAHE3D)
    cudaFree(dev_ptr);
    cudaFree(dev_out);
    clahe3DOut = std::move(out);
  }
  */
  /*
  cv::namedWindow(OUTPUT_WINDOW_NAME, cv::WINDOW_KEEPRATIO);

  for (size_t i = 0; i < frames.size(); ++i) {
    std::vector<cv::Mat> m = { frames[i],
                               equalizeHistOut.empty() ? frames[i] : equalizeHistOut[i],
                               clahe2DOut.empty() ? frames[i] : clahe2DOut[i],
                               clahe3DOut.empty() ? frames[i] : clahe3DOut[i] };

    for (int i = 0; i < m.size(); ++i) {
      cv::Mat tmp;
      double factor = double(OUTPUT_IMAGE_HEIGHT) / double(m[i].rows);
      cv::resize(m[i], tmp, cv::Size(), factor, factor);
      m[i] = tmp;
    }

    drawGrid(m[2], cv::Size(numTiles, numTiles));
    drawGrid(m[3], cv::Size(numTiles, numTiles));

    cv::Mat output = concat(m[0], m[1], m[2], m[3]);

    cv::imshow(OUTPUT_WINDOW_NAME, output);
    if ((cv::waitKey(0) & 0xff) == 27) {
      break;
    }
  }*/

  return EXIT_SUCCESS;
}
