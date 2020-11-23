#ifndef __NIFTI_HPP
#define __NIFTI_HPP

#include <opencv2/core.hpp>

namespace nifti {

    std::vector<cv::Mat> read(const std::string &fname);

}

#endif // __NIFTI_HPP
