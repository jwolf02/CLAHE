#ifndef __CLAHE3D_HPP
#define __CLAHE3D_HPP

#include <vector>
#include <opencv2/opencv.hpp>

namespace cv {

    template <typename T>
    class Size3_ {
    public:

        Size3_() = default;

        Size3_(T height, T width, T depth) : height(height), width(width), depth(depth) {}

        Size3_(const Size3_ &sz) : height(sz.height), width(sz.width), depth(sz.depth) {}

        explicit Size3_(const Point3_<T> &p) : height(p.y), width(p.x), depth(p.z) {}

        Size3_& operator=(const Size3_ &sz) {
          height = sz.height;
          width = sz.width;
          depth = sz.depth;
          return *this;
        }

        double aspectRatio() const {
          return double(height) / width;
        }

        T area() const {
          return height * width;
        }

        T volume() const {
          return height * width * depth;
        }

        bool empty() const {
          return height <= 0 || width <= 0 || depth <= 0;
        }

        template <typename T2>
        explicit operator Size3_<T2>() const {
          return Size3_<T2>(height, width, depth);
        }

        T height = 0;

        T width = 0;

        T depth = 0;
    };

    typedef Size3_<float> Size3f;
    typedef Size3_<int> Size3i;

    //! @} imgproc_shape

    //! @addtogroup imgproc_hist
    //! @{

    /** @brief Base class for Contrast Limited Adaptive Histogram Equalization in 3D.
    */
    class CV_EXPORTS_W CLAHE3D : public Algorithm {
    public:

        CV_WRAP virtual void apply(const std::vector<cv::Mat> &, std::vector<cv::Mat> &) = 0;

        CV_WRAP virtual void setClipLimit(double clipLimit) = 0;

        CV_WRAP virtual double getClipLimit() const = 0;

        CV_WRAP virtual void setTilesGridSize(Size3i tileGridSize) = 0;

        CV_WRAP virtual Size3i getTilesGridSize() const = 0;

        CV_WRAP virtual void collectGarbage() = 0;

    };

    /** @brief Creates a smart pointer to a cv::CLAHE3D class and initializes it.

    @param clipLimit Threshold for contrast limiting.
    @param tileGridSize Size3i of grid for histogram equalization. Input image vector will be divided into
    equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column and the number of frames.
     */
    CV_EXPORTS_W Ptr<CLAHE3D> createCLAHE3D(double clipLimit = 40.0, Size3i tilesGridSize = Size3i(8, 8, 8));

}

#endif // __CLAHE3D_HPP
