/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined CUDA_DISABLER

#include <opencv2/cudev.hpp>
#include "clahe.cuh"
#include "cuda_helpers.cuh"
#include <opencv2/cudaarithm.hpp>

using namespace cv::cudev;

texture<float, cudaTextureType3D> tex;

namespace clahe
{
    void calcLut_8U(PtrStepSzb src, PtrStep<float> lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, cudaStream_t stream);
    void calcLut_16U(PtrStepSzus src, PtrStepus lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, PtrStepSzi hist, cudaStream_t stream);
    template <typename T> void transform(PtrStepSz<T> src, PtrStepSz<T> dst, int tilesX, int tilesY, int2 tileSize, cudaStream_t stream);
}

namespace
{
    class CLAHE_Impl : public CLAHE2D
    {
    public:
        explicit CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);

        ~CLAHE_Impl() override { CV_CUDEV_SAFE_CALL(cudaFreeArray(array_)); }

        void apply(cv::InputArray src, cv::OutputArray dst) override;
        void apply(cv::InputArray src, cv::OutputArray dst, Stream& stream) override;

        void setClipLimit(double clipLimit) override;
        double getClipLimit() const override;

        void setTilesGridSize(cv::Size tileGridSize) override;
        cv::Size getTilesGridSize() const override;

        void collectGarbage() override;

    private:
        double clipLimit_;
        int tilesX_;
        int tilesY_;

        GpuMat srcExt_;
        GpuMat lut_;
        GpuMat hist_; // histogram on global memory for CV_16UC1 case

        cudaArray_t array_;
    };

    CLAHE_Impl::CLAHE_Impl(double clipLimit, int tilesX, int tilesY) :
            clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY) {


        ensureSizeIsEnough(tilesX_ * tilesY_, 256, CV_32F, lut_);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        array_ = nullptr;
        cudaExtent ext = make_cudaExtent(256, tilesX_, tilesY_);
        CV_CUDEV_SAFE_CALL(cudaMalloc3DArray(&array_, &channelDesc, ext));

         /*
        // and copy image data
        cudaMemcpy3DParms myparms = { nullptr };
        myparms.srcPos = make_cudaPos(0,0,0);
        myparms.dstPos = make_cudaPos(0,0,0);
        // The source pointer must be a cuda pitched ptr.
        myparms.srcPtr = make_cudaPitchedPtr(lut_.data, ext.width * sizeof(float), ext.width, ext.height);
        myparms.dstArray = array_;
        myparms.extent = ext;
        myparms.kind = cudaMemcpyDeviceToDevice;
        CV_CUDEV_SAFE_CALL(cudaMemcpy3D(&myparms));
        */

        // Set texture parameters
        tex.addressMode[0] = cudaAddressModeClamp;
        tex.addressMode[1] = cudaAddressModeClamp;
        tex.addressMode[2] = cudaAddressModeClamp;
        tex.filterMode = cudaFilterModeLinear;
        tex.normalized = true;    // access with normalized texture coordinates

        // Bind the array to the texture
        CV_CUDEV_SAFE_CALL(cudaBindTextureToArray(tex, array_, channelDesc));
    }

    void CLAHE_Impl::apply(cv::InputArray _src, cv::OutputArray _dst) {
        apply(_src, _dst, Stream::Null());
    }

    void CLAHE_Impl::apply(cv::InputArray _src, cv::OutputArray _dst, Stream& s) {
        GpuMat src = _src.getGpuMat();

        const int type = src.type();

        CV_Assert( type == CV_8UC1 || type == CV_16UC1 );

        _dst.create( src.size(), type );
        GpuMat dst = _dst.getGpuMat();

        const int histSize = type == CV_8UC1 ? 256 : 65536;

        ensureSizeIsEnough(tilesX_ * tilesY_, histSize, CV_32F, lut_);

        cudaStream_t stream = StreamAccessor::getStream(s);

        cv::Size tileSize;
        GpuMat srcForLut;

        if (src.cols % tilesX_ == 0 && src.rows % tilesY_ == 0) {
            tileSize = cv::Size(src.cols / tilesX_, src.rows / tilesY_);
            srcForLut = src;
        } else {
#ifndef HAVE_OPENCV_CUDAARITHM
            throw_no_cuda();
#else
            cv::cuda::copyMakeBorder(src, srcExt_, 0, tilesY_ - (src.rows % tilesY_), 0, tilesX_ - (src.cols % tilesX_), cv::BORDER_REFLECT_101, cv::Scalar(), s);
#endif

            tileSize = cv::Size(srcExt_.cols / tilesX_, srcExt_.rows / tilesY_);
            srcForLut = srcExt_;
        }

        const int tileSizeTotal = tileSize.area();
        const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

        int clipLimit = 0;
        if (clipLimit_ > 0.0) {
            clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
            clipLimit = std::max(clipLimit, 1);
        }

        if (type == CV_8UC1)
            clahe::calcLut_8U(srcForLut, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), clipLimit, lutScale, stream);
        else { CV_Assert(!!(false)); }

        cudaExtent ext = make_cudaExtent(256, tilesX_, tilesY_);
        /*
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        array_ = nullptr;
        CV_CUDEV_SAFE_CALL(cudaMalloc3DArray(&array_, &channelDesc, ext));
        */

        // and copy image data
        cudaMemcpy3DParms myparms = { nullptr };
        myparms.srcPos = make_cudaPos(0,0,0);
        myparms.dstPos = make_cudaPos(0,0,0);
        // The source pointer must be a cuda pitched ptr.
        myparms.srcPtr = make_cudaPitchedPtr(lut_.data, ext.width * sizeof(float), ext.width, ext.height);
        myparms.dstArray = array_;
        myparms.extent = ext;
        myparms.kind = cudaMemcpyDeviceToDevice;
        CV_CUDEV_SAFE_CALL(cudaMemcpy3D(&myparms));

        /*
        // Set texture parameters
        tex.addressMode[0] = cudaAddressModeClamp;
        tex.addressMode[1] = cudaAddressModeClamp;
        tex.addressMode[2] = cudaAddressModeClamp;
        tex.filterMode = cudaFilterModeLinear;
        tex.normalized = true;    // access with normalized texture coordinates

        // Bind the array to the texture
        CV_CUDEV_SAFE_CALL(cudaBindTextureToArray(tex, array_, channelDesc));
        */

        if (type == CV_8UC1) {
            clahe::transform<uchar>(src, dst, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), stream);
        } else { // type == CV_16UC1
            clahe::transform<ushort>(src, dst, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), stream);
        }
    }

    void CLAHE_Impl::setClipLimit(double clipLimit)
    {
        clipLimit_ = clipLimit;
    }

    double CLAHE_Impl::getClipLimit() const
    {
        return clipLimit_;
    }

    void CLAHE_Impl::setTilesGridSize(cv::Size tileGridSize)
    {
        tilesX_ = tileGridSize.width;
        tilesY_ = tileGridSize.height;
    }

    cv::Size CLAHE_Impl::getTilesGridSize() const
    {
        return cv::Size(tilesX_, tilesY_);
    }

    void CLAHE_Impl::collectGarbage()
    {
        srcExt_.release();
        lut_.release();
        CV_CUDEV_SAFE_CALL(cudaFreeArray(array_));
    }
}

cv::Ptr<CLAHE2D> createCLAHE2D(double clipLimit, const cv::Size &tileGridSize)
{
    return cv::makePtr<CLAHE_Impl>(clipLimit, tileGridSize.width, tileGridSize.height);
}

namespace clahe
{
    GLOBALQUALIFIER
    void calcLutKernel_8U(const PtrStepb src, PtrStep<float> lut,
                                     const int2 tileSize, const int tilesX,
                                     const int clipLimit, const float lutScale) {
        __shared__ int smem[256];

        const int tx = blockIdx.x;
        const int ty = blockIdx.y;
        const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        smem[tid] = 0;
        __syncthreads();

        for (int i = threadIdx.y; i < tileSize.y; i += blockDim.y) {
            const uchar* srcPtr = src.ptr(ty * tileSize.y + i) + tx * tileSize.x;
            for (int j = threadIdx.x; j < tileSize.x; j += blockDim.x) {
                const int data = srcPtr[j];
                ::atomicAdd(&smem[data], 1);
            }
        }

        __syncthreads();

        int tHistVal = smem[tid];

        __syncthreads();

        if (clipLimit > 0) {
            // clip histogram bar
            /*
            int clipped = 0;
            if (tHistVal > clipLimit) {
                clipped = tHistVal - clipLimit;
                tHistVal = clipLimit;
            }
            */
            const int32_t temp = tHistVal - clipLimit;
            const auto s = (temp & (1U << 31)) >> 31;
            int clipped = (1 - s) * temp;
            tHistVal = s * tHistVal + (1 - s) * clipLimit;

            // find number of overall clipped samples
            blockReduce<256>(smem, clipped, tid, plus<int>());

            // broadcast evaluated value
            __shared__ int totalClipped;
            __shared__ int redistBatch;
            __shared__ int residual;
            __shared__ int rStep;

            if (tid == 0) {
                totalClipped = clipped;
                redistBatch = totalClipped / 256;
                residual = totalClipped & 0xff;//- redistBatch * 256;
                rStep = residual != 0 ? 256 / residual : 1;
            }
            __syncthreads();

            // redistribute clipped samples evenly
            tHistVal += redistBatch;

            if (residual && tid % rStep == 0 && tid / rStep < residual) {
                tHistVal += 1;
            }
        }

        const auto lutVal = static_cast<float>(blockScanInclusive<256>(tHistVal, smem, tid));

        lut(ty * tilesX + tx, (int) tid) = lutScale * lutVal;
    }

    void calcLut_8U(PtrStepSzb src, PtrStep<float> lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, cudaStream_t stream) {
        const dim3 block(32, 8);
        const dim3 grid(tilesX, tilesY);

        calcLutKernel_8U<<<grid, block, 0, stream>>>(src, lut, tileSize, tilesX, clipLimit, lutScale);

        CV_CUDEV_SAFE_CALL(cudaGetLastError());

        if (stream == nullptr) {
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
        }
    }

    template <typename T>
    __global__ void transformKernel(const PtrStepSz<T> src, PtrStepSz<T> dst, const int2 tileSize, const int tilesX, const int tilesY)
    {
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < src.cols && y < src.rows) {
            const auto grayValue = (src(y, x) + 0.5f) / 255.0f;
            const auto xPos = static_cast<float>(x) / static_cast<float>(tileSize.x * tilesX);
            const auto yPos = static_cast<float>(y) / static_cast<float>(tileSize.y * tilesY);
            dst(y, x) = saturate_cast<T>(tex3D(tex, grayValue, xPos, yPos));
        }
    }

    template <typename T>
    void transform(PtrStepSz<T> src, PtrStepSz<T> dst, int tilesX, int tilesY, int2 tileSize, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        CV_CUDEV_SAFE_CALL( cudaFuncSetCacheConfig(transformKernel<T>, cudaFuncCachePreferL1) );

        transformKernel<T><<<grid, block, 0, stream>>>(src, dst, tileSize, tilesX, tilesY);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );


        if (stream == 0) {
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
        }
    }
}

#endif // CUDA_DISABLER
