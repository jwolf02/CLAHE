#ifndef __CLAHE3D_CUH
#define __CLAHE3D_CUH

#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda_types.hpp>

namespace cv {
    namespace cuda {

        class CLAHE3D : public cv::CLAHE3D {
        public:

            using cv::CLAHE3D::apply;

            virtual void apply(const std::vector<cv::Mat> &, std::vector<cv::Mat> &, cv::cuda::Stream &) = 0;

            virtual void
            apply(cv::cuda::DevPtr<uchar> src, cv::cuda::DevPtr<uchar> dest, int rows, int cols, int frames,
                  cv::cuda::Stream &) = 0;
        };

        Ptr<cv::CLAHE3D> createCLAHE3D(double clipLimit = 40.0, Size3i grid = Size3i(8, 8, 8));
    }
}

#endif // __CLAHE3D_CUH
