#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#include "nifti.hpp"
#include <cuda_helpers.cuh>

template <typename dtype>
static std::vector<cv::Mat> read_slices(std::ifstream &is, const int &num_slices, const int &w, const int &h) {
  const int buffer_size = num_slices * w * h;

  std::unique_ptr<dtype[]>data(new dtype[buffer_size]);
  is.read(reinterpret_cast<char*>(data.get()), sizeof(dtype) * buffer_size);

  std::vector<cv::Mat> slices(num_slices);

  for (int i = 0; i < num_slices; ++i) {
    slices[i] = cv::Mat(w, h, CV_32SC1);
    int *img_data = (int *) slices[i].data;

    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < h; ++k) {
        const int idx1 = j * h + k;
        const int idx2 = (k * w + j) * num_slices + i;
        img_data[idx1] = data[idx2];
      }
    }
    cv::transpose(slices[i], slices[i]);
  }

  return slices;
}

std::vector<cv::Mat> nifti::read(const std::string &fname) {
  int header_size;
  short int num_slices;
  short int w, h;
  short int dtype;

  std::ifstream is(fname, std::ios::binary | std::ios::in);
  if (!is) {
    std::cerr << "invalid filename" << std::endl;
    std::exit(1);
  }

  is.read(reinterpret_cast<char*>(&header_size), sizeof(int));
  is.seekg(0);
  is.seekg(42);

  is.read(reinterpret_cast<char*>(&w), sizeof(short int));
  is.read(reinterpret_cast<char*>(&h), sizeof(short int));
  is.read(reinterpret_cast<char*>(&num_slices), sizeof(short int));

  is.seekg(0);
  is.seekg(70);
  is.read(reinterpret_cast<char*>(&dtype), sizeof(short int));

  is.seekg(0);
  is.seekg(header_size);

  std::vector<cv::Mat> slices(num_slices);

  //std::cout << num_slices << " " << w << " " << h << std::endl;

  switch (dtype) {
    case 4: // short int
      slices = read_slices<short>(is, num_slices, w, h);
      break;
    case 16: // float
      slices = read_slices<float>(is, num_slices, w, h);
      break;
    case 64: // long long
      slices = read_slices<long long>(is, num_slices, w, h);
      break;
    default:{
      std::cerr << "Parser type not understood!" << std::endl
                << "Please see the documentation at https://brainder.org/2012/09/23/the-nifti-file-format/ "
                << "and add a new item in the switch format according to the found value (" << dtype << ")" << std::endl;
    } break;
      // add more cases here according to your needs
  }

  is.close();

  return slices;
}