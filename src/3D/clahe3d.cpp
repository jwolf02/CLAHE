#include <opencv2/opencv.hpp>
#include <vector>
#include <3D/clahe3d.hpp>
#include <iostream>
#include <numeric>

using namespace cv;

typedef Size3i int3;
static inline int3 make_int3(int x, int y, int z) { return { x, y, z }; }

class CLAHE3D_Impl : public CLAHE3D {
public:

    CLAHE3D_Impl() = default;

    CLAHE3D_Impl(double clipLimit, const Size3i &tilesGridSize);

    void apply(const std::vector<cv::Mat> &in, std::vector<cv::Mat> &out) override;

    void setClipLimit(double clipLimit) override;

    double getClipLimit() const override;

    void setTilesGridSize(Size3i tileGridSize) override;

    Size3i getTilesGridSize() const override;

    void collectGarbage() override;

private:

    double _clipLimit = 0.0;

    cv::Mat _lut;

    Size3i _tiles;

};

void CLAHE3D_Impl::setClipLimit(double clipLimit) { _clipLimit = clipLimit; }

double CLAHE3D_Impl::getClipLimit() const { return _clipLimit; }

void CLAHE3D_Impl::setTilesGridSize(Size3i tileGridSize) { _tiles = tileGridSize; }

Size3i CLAHE3D_Impl::getTilesGridSize() const { return _tiles; }

void CLAHE3D_Impl::collectGarbage() { _lut.release(); }

cv::Ptr<CLAHE3D> createCLAHE3D(double clipLimit, const Size3i &tilesGridSize) {
  return new CLAHE3D_Impl(clipLimit, tilesGridSize);
}

static void calcLut(const std::vector<cv::Mat> &in, cv::Mat &lut, int clipLimit, int3 tileSize, int tilesX, int tilesY, int tilesZ);
static void transform(const std::vector<cv::Mat> &in, std::vector<cv::Mat> &out, const cv::Mat &lut, int3 tileSize, int tilesX, int tilesY, int tilesZ);

CLAHE3D_Impl::CLAHE3D_Impl(double clipLimit, const Size3i &tilesGridSize) : _clipLimit(clipLimit), _tiles(tilesGridSize) {}

void CLAHE3D_Impl::apply(const std::vector<cv::Mat> &in, std::vector<cv::Mat> &out) {
  CV_Assert(in.size() >= _tiles.z);

  int3 tileSize = make_int3(in[0].cols / _tiles.x, in[0].rows / _tiles.y, in.size() / _tiles.z);
  int tileSizeTotal = tileSize.x * tileSize.y * tileSize.z;

  int clipLimit = 0;
  if (_clipLimit > 0.0) {
    clipLimit = static_cast<int>(_clipLimit * tileSizeTotal / 256);
    clipLimit = std::max(clipLimit, 1);
  }

  const int volume = _tiles.x * _tiles.y * _tiles.z;

  cv::cuda::ensureSizeIsEnough(volume, 256, CV_8UC1, _lut);

  calcLut(in, _lut, clipLimit, tileSize, _tiles.x, _tiles.y, _tiles.z);
  transform(in, out, _lut, tileSize, _tiles.x, _tiles.y, _tiles.z);
}

static inline int reflect101(int x, int xmax) {
  return std::min(x, xmax) - std::max(0, x - xmax);
}

void calcLut(const std::vector<cv::Mat> &in, cv::Mat &lut, int clipLimit, int3 tileSize, int tilesX, int tilesY, int tilesZ) {
  const float lutScale = 255.0f / float(tileSize.x * tileSize.y * tileSize.z);
  for (int tz = 0; tz < tilesZ; ++tz) {
    for (int ty = 0; ty < tilesY; ++ty) {
      for (int tx = 0; tx < tilesX; ++tx) {

        uint32_t smem[256];
        for (int i = 0; i < 256; ++i) {
          smem[i] = 0;
        }

        // compute histogram
        for (int i = 0; i < tileSize.z; ++i) {
          const auto &mat = in[reflect101(i, in.size() - 1)];
          for (int j = 0; j < tileSize.y; ++j) {
            const uchar* srcPtr = mat.ptr(reflect101((int) (ty * tileSize.y + j), mat.rows - 1));
            for (int k = 0; k < tileSize.x; ++k) {
              const auto val = srcPtr[reflect101((int) tx * tileSize.x + k, mat.cols - 1)];
              smem[val]++;
            }
          }
        }

        // compute total number of clipped examples
        int totalClipped = 0;
        for (int i = 0; i < 256; ++i) {
          int clipped = 0;
          if (smem[i] > clipLimit) {
            clipped = (int) (smem[i] - clipLimit);
            smem[i] = clipLimit;
          }
          totalClipped += clipped;
        }

        // redistribute residue
        int redistBatch = totalClipped / 256;
        int residual = totalClipped & 0xff;//- redistBatch * 256;
        int rStep = residual != 0 ? 256 / residual : 1;

        for (int i = 0; i < 256; ++i) {
          smem[i] += redistBatch;

          if (residual && i % rStep == 0 && i / rStep < residual) {
            smem[i] += 1;
          }
        }

        // equalize histogram
        for (int i = 1; i < 256; ++i) {
          smem[i] += smem[i - 1];
        }

        auto ptr = lut.ptr((tz * tilesY + ty) * tilesX + tx);
        for (int i = 0; i < 256; ++i) {
          ptr[i] = lutScale * static_cast<float>(smem[i]);
        }
      }
    }
  }
}

void transform(const std::vector<cv::Mat> &in, std::vector<cv::Mat> &out, const cv::Mat &lut, int3 tileSize, int tilesX, int tilesY, int tilesZ) {
  out.clear();
  for (int k = 0; k < in.size(); ++k) {
    cv::Mat o(in[k].size(), in[k].type());
    CV_Assert(in[k].rows == in[std::min<int>(k + 1, in.size() - 1)].rows && in[k].cols == in[std::min<int>(k + 1, in.size() - 1)].cols);
    for (int j = 0; j < o.rows; ++j) {
      auto srcPtr = in[k].ptr(j);
      auto dstPtr = o.ptr(j);
      for (int i = 0; i < o.cols; ++i) {

        const float tzf = (static_cast<float>(k) / tileSize.z) - 0.5f;
        int tz1 = int(tzf);
        int tz2 = tz1 + 1;
        const float za = tzf - static_cast<float>(tz1);
        tz1 = std::max(tz1, 0);
        tz2 = std::min(tz2, tilesY - 1);

        const float tyf = (static_cast<float>(j) / tileSize.y) - 0.5f;
        int ty1 = int(tyf);
        int ty2 = ty1 + 1;
        const float ya = tyf - static_cast<float>(ty1);
        ty1 = std::max(ty1, 0);
        ty2 = std::min(ty2, tilesY - 1);

        const float txf = (static_cast<float>(i) / tileSize.x) - 0.5f;
        int tx1 = int(txf);
        int tx2 = tx1 + 1;
        const float xa = txf - static_cast<float>(tx1);
        tx1 = std::max(tx1, 0);
        tx2 = std::min(tx2, tilesX - 1);

        const int srcVal = srcPtr[i];

        float res = 0.0f;

        res += static_cast<float>(lut.ptr((tz1 * tilesY + ty1) * tilesX + tx1)[srcVal]) * ((1.0f - za) * (1.0f - ya) * (1.0f - xa));
        res += static_cast<float>(lut.ptr((tz1 * tilesY + ty1) * tilesX + tx2)[srcVal]) * ((1.0f - za) * (1.0f - ya) * (xa));
        res += static_cast<float>(lut.ptr((tz1 * tilesY + ty2) * tilesX + tx1)[srcVal]) * ((1.0f - za) * (ya) * (1.0f - xa));
        res += static_cast<float>(lut.ptr((tz1 * tilesY + ty2) * tilesX + tx2)[srcVal]) * ((1.0f - za) * (ya) * (xa));
        res += static_cast<float>(lut.ptr((tz2 * tilesY + ty1) * tilesX + tx1)[srcVal]) * (za * (1.0f - ya) * (1.0f - xa));
        res += static_cast<float>(lut.ptr((tz2 * tilesY + ty1) * tilesX + tx2)[srcVal]) * (za * (1.0f - ya) * (xa));
        res += static_cast<float>(lut.ptr((tz2 * tilesY + ty2) * tilesX + tx1)[srcVal]) * (za * (ya) * (1.0f - xa));
        res += static_cast<float>(lut.ptr((tz2 * tilesY + ty2) * tilesX + tx2)[srcVal]) * (za * (ya) * (xa));

        dstPtr[i] = cv::saturate_cast<uchar>(res);
      }
    }

    out.push_back(std::move(o));
  }
}
