#ifndef __CLAHE_CUH
#define __CLAHE_CUH

#include <opencv2/imgproc.hpp>

class CLAHE2D : public cv::CLAHE {
public:
    using cv::CLAHE::apply;
    /** @brief Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.

    @param src Source image with CV_8UC1 type.
    @param dst Destination image.
    @param stream Stream for the asynchronous version.
     */
    CV_WRAP virtual void apply(cv::InputArray src, cv::OutputArray dst, cv::cuda::Stream& stream) = 0;
};

/** @brief Creates implementation for cuda::CLAHE .

@param clipLimit Threshold for contrast limiting.
@param tileGridSize Size of grid for histogram equalization. Input image will be divided into
equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.
 */
cv::Ptr<CLAHE2D> createCLAHE2D(double clipLimit = 40.0, const cv::Size &tileGridSize = cv::Size(8, 8));

#endif // __CLAHE_CUH
