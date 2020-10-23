#ifndef __CLAHE3D_HPP
#define __CLAHE3D_HPP

#include <vector>
#include <opencv2/opencv.hpp>

namespace cv {

    typedef Point3i Size3i;

    class CV_EXPORTS_W CLAHE3D : public Algorithm {
    public:

        CV_WRAP virtual void apply(const std::vector<cv::Mat> &, std::vector<cv::Mat> &) = 0;

        CV_WRAP virtual void setClipLimit(double clipLimit) = 0;

        CV_WRAP virtual double getClipLimit() const = 0;

        CV_WRAP virtual void setTilesGridSize(Size3i tileGridSize) = 0;

        CV_WRAP virtual Size3i getTilesGridSize() const = 0;

        CV_WRAP virtual void collectGarbage() = 0;

    };

    CV_EXPORTS_W Ptr<CLAHE3D> createCLAHE3D(double clipLimit = 40.0, Size3i tilesGridSize = Size3i(8, 8, 8));

}

#endif // __CLAHE3D_HPP
