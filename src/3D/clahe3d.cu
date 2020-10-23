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

#include <clahe3d.cuh>
#include <cuda_helpers.cuh>
#include <opencv2/cudev.hpp>

using namespace cv;
using cv::cuda::PtrStepSz;
using namespace cv::cudev;

static void calcLut(PtrStep<uchar> src, PtrStep<uchar> lut, int3 tiles, int3 tileSize, int3 inputSize, int clipLimit, float lutScale, cudaStream_t stream);
static void transform(PtrStep<uchar> src, PtrStep<uchar> dest, PtrStep<uchar> lut, int3 tiles, int3 tileSize, int3 inputSize, cudaStream_t stream);

namespace
{
class CLAHE3D_Impl : public cv::cuda::CLAHE3D {
    public:

        explicit CLAHE3D_Impl(double clipLimit, Size3i tilesGridSize);

        void apply(const std::vector<cv::Mat> &in, std::vector<cv::Mat> &out) override;

        void apply(const std::vector<cv::Mat> &in, std::vector<cv::Mat> &out, cv::cuda::Stream &stream) override;

        void apply(DevPtr<uchar> src, DevPtr<uchar> dest, int rows, int cols, int frames, cv::cuda::Stream &stream) override;

        void setClipLimit(double clipLimit) override;

        double getClipLimit() const override;

        void setTilesGridSize(Size3i tileGridSize) override;

        Size3i getTilesGridSize() const override;

        void collectGarbage() override;

    private:

        double _clipLimit = 0;

        cv::cuda::GpuMat _lut;

        Size3i _tiles;

    };
}

cv::Ptr<cv::CLAHE3D> cv::cuda::createCLAHE3D(double clipLimit, Size3i grid) {
  return new CLAHE3D_Impl(clipLimit, grid);
}

CLAHE3D_Impl::CLAHE3D_Impl(double clipLimit, Size3i tilesGridSize) : _clipLimit(clipLimit), _tiles(std::move(tilesGridSize)) {}

void CLAHE3D_Impl::setClipLimit(double clipLimit) { _clipLimit = clipLimit; }

double CLAHE3D_Impl::getClipLimit() const { return _clipLimit; }

void CLAHE3D_Impl::setTilesGridSize(Size3i tileGridSize) { _tiles = tileGridSize; }

Size3i CLAHE3D_Impl::getTilesGridSize() const { return _tiles; }

void CLAHE3D_Impl::collectGarbage() { _lut.release(); }

void CLAHE3D_Impl::apply(const std::vector<cv::Mat> &in, std::vector<cv::Mat> &out) {
  return apply(in, out, cv::cuda::Stream::Null());
}

void CLAHE3D_Impl::apply(const std::vector<cv::Mat> &in, std::vector<cv::Mat> &out, cv::cuda::Stream &stream) {
  cv::cuda::GpuMat src;
  cv::cuda::GpuMat dest;

  int rows = in[0].rows;
  int cols = in[0].cols;
  int frames = in.size();

  ensureSizeIsEnough(rows * frames, cols, CV_8UC1, src);
  ensureSizeIsEnough(rows * frames, cols, CV_8UC1, dest);

  for (int i = 0; i < frames; ++i) {
    CV_Assert(in[i].rows == rows && in[i].cols == cols);
    auto tmp = src(cv::Range(rows * i, rows * (i + 1)), cv::Range(0, cols));
    tmp.upload(in[i]);
  }

  apply(src.data, dest.data, rows, cols, frames, stream);

  out.clear();
  for (int i = 0; i < frames; ++i) {
    auto tmp = dest(cv::Range(rows * i, rows * (i + 1)), cv::Range(0, cols));
    cv::Mat m;
    tmp.download(m);
    out.push_back(m);
  }
}

void CLAHE3D_Impl::apply(DevPtr<uchar> src, DevPtr<uchar> dest, int rows, int cols, int frames, cv::cuda::Stream &stream) {
  const int volume = _tiles.x * _tiles.y * _tiles.z;
  ensureSizeIsEnough(volume, 256, CV_8UC1, _lut);

  int3 tileSize = make_int3(cols / _tiles.x, rows / _tiles.y, frames / _tiles.z);
  int tileSizeTotal = tileSize.x * tileSize.y * tileSize.z;
  const int histSize = 256;
  const float lutScale = static_cast<float>(histSize - 1) / static_cast<float>(tileSizeTotal);
  int3 tiles = make_int3(_tiles.x, _tiles.y, _tiles.z);
  int3 inputSize = make_int3(cols, rows, frames);

  int clipLimit = 0;
  if (_clipLimit > 0.0) {
    clipLimit = std::max(static_cast<int>(_clipLimit * tileSizeTotal / histSize), 1);
  }

  cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);

  calcLut(PtrStep<uchar>(src.data, cols), _lut, tiles, tileSize, inputSize, clipLimit, lutScale, s);
  transform(PtrStep<uchar>(src.data, cols), PtrStep<uchar>(dest, cols), _lut, tiles, tileSize, inputSize, s);
}

DEVICEQUALIFIER INLINEQUALIFIER
int reflect101(int x, int xmax) {
  return min(x, xmax) - max(0, x - xmax);
}

GLOBALQUALIFIER
void calcLut_kernel(PtrStep<uchar> src, PtrStep<uchar> lut, int3 tiles, int3 tileSize, int3 inputSize, int clipLimit, float lutScale) {
  __shared__ int smem[256];

  const uint64_t tx = blockIdx.x;
  const uint64_t ty = blockIdx.y;
  const uint64_t tz = blockIdx.z;
  const uint64_t tid = threadIdx.y * blockDim.x + threadIdx.x;

  smem[tid] = 0;
  __syncthreads();

  for (uint64_t i = tz * tileSize.z; i < (tz + 1) * tileSize.z; ++i) {
    const int ri = reflect101((int) i, inputSize.z - 1);
    for (uint64_t j = threadIdx.y; j < tileSize.y; j += blockDim.y) {
      const uchar *srcPtr = src.ptr(ri * tiles.x + reflect101(ty * tileSize.y + j, inputSize.x - 1));
      for (uint64_t k = threadIdx.x; k < tileSize.x; k += blockDim.x) {
        const int data = srcPtr[reflect101(tx * tileSize.x + k, inputSize.y - 1)];
        ::atomicAdd(&smem[data], 1);
      }
    }
  }

  __syncthreads();

  int tHistVal = smem[tid];

  __syncthreads();

  if (clipLimit > 0) {
    // clip histogram bar
    int clipped = 0;
    if (tHistVal > clipLimit) {
      clipped = tHistVal - clipLimit;
      tHistVal = clipLimit;
    }
    __syncthreads();

    // find number of overall clipped samples
    blockReduce<256>(smem, clipped, tid, plus<int>());

    // broadcast evaluated value
    __shared__ uint32_t redistBatch;
    __shared__ uint32_t residual;
    __shared__ uint32_t rStep;

    if (tid == 0) {
      uint32_t totalClipped = clipped;
      redistBatch = totalClipped / 256;
      residual = totalClipped & 0xffU;//- redistBatch * 256;
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

  lut(static_cast<uint32_t>((tz * tiles.y + ty) * tiles.x + tx), (int) tid) = lutScale * lutVal;
}

void calcLut(PtrStep<uchar> src, PtrStep<uchar> lut, int3 tiles, int3 tileSize, int3 inputSize, int clipLimit, float lutScale, cudaStream_t stream) {
  const dim3 block(32, 8, 1);
  const dim3 grid(tiles.x, tiles.y, tiles.z);

  calcLut_kernel<<<grid, block, 0, stream>>>(src, lut, tiles, tileSize, inputSize, clipLimit, lutScale);
  CV_CUDEV_SAFE_CALL(cudaGetLastError());

  if (stream == nullptr) {
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
  }
}

GLOBALQUALIFIER
void transform_kernel(PtrStep<uchar> src, PtrStep<uchar> dest, PtrStep<uchar> lut, int3 tiles, int3 inputSize, int3 tileSize) {
  const uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inputSize.x && y < inputSize.y && z < inputSize.z) {
    const float tzf = (static_cast<float>(z) / tileSize.z) - 0.5f;
    int tz1 = int(tzf);
    int tz2 = tz1 + 1;
    const float za = tzf - static_cast<float>(tz1);
    tz1 = max(tz1, 0);
    tz2 = min(tz2, tiles.z - 1);

    const float tyf = (static_cast<float>(y) / tileSize.y) - 0.5f;
    int ty1 = int(tyf);
    int ty2 = ty1 + 1;
    const float ya = tyf - static_cast<float>(ty1);
    ty1 = max(ty1, 0);
    ty2 = min(ty2, tiles.y - 1);

    const float txf = (static_cast<float>(x) / tileSize.x) - 0.5f;
    int tx1 = int(txf);
    int tx2 = tx1 + 1;
    const float xa = txf - static_cast<float>(tx1);
    tx1 = max(tx1, 0);
    tx2 = min(tx2, tiles.x - 1);

    const int srcVal = src(z * inputSize.y + y, x);

    float res = 0.0f;

    res += static_cast<float>(lut.ptr((tz1 * tiles.y + ty1) * tiles.x + tx1)[srcVal]) * ((1.0f - za) * (1.0f - ya) * (1.0f - xa));
    res += static_cast<float>(lut.ptr((tz1 * tiles.y + ty1) * tiles.x + tx2)[srcVal]) * ((1.0f - za) * (1.0f - ya) * (xa));
    res += static_cast<float>(lut.ptr((tz1 * tiles.y + ty2) * tiles.x + tx1)[srcVal]) * ((1.0f - za) * (ya) * (1.0f - xa));
    res += static_cast<float>(lut.ptr((tz1 * tiles.y + ty2) * tiles.x + tx2)[srcVal]) * ((1.0f - za) * (ya) * (xa));
    res += static_cast<float>(lut.ptr((tz2 * tiles.y + ty1) * tiles.x + tx1)[srcVal]) * (za * (1.0f - ya) * (1.0f - xa));
    res += static_cast<float>(lut.ptr((tz2 * tiles.y + ty1) * tiles.x + tx2)[srcVal]) * (za * (1.0f - ya) * (xa));
    res += static_cast<float>(lut.ptr((tz2 * tiles.y + ty2) * tiles.x + tx1)[srcVal]) * (za * (ya) * (1.0f - xa));
    res += static_cast<float>(lut.ptr((tz2 * tiles.y + ty2) * tiles.x + tx2)[srcVal]) * (za * (ya) * (xa));

    dest(z * inputSize.y + y, x) = cv::cudev::saturate_cast<uchar>(res);
  }
}

void transform(PtrStep<uchar> src, PtrStep<uchar> dest, PtrStep<uchar> lut, int3 tiles, int3 tileSize, int3 inputSize, cudaStream_t stream) {
  const dim3 block(32, 8, 4);
  const dim3 grid(divUp(inputSize.x, block.x), divUp(inputSize.y, block.y), divUp(inputSize.z, block.z));

  CV_CUDEV_SAFE_CALL(cudaFuncSetCacheConfig(transform_kernel, cudaFuncCachePreferL1));

  transform_kernel<<<grid, block, 0, stream>>>(src, dest, lut, tiles, inputSize, tileSize);
  CV_CUDEV_SAFE_CALL(cudaGetLastError());

  if (stream == nullptr) {
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
  }
}

#endif
