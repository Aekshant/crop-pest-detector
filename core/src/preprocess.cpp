#include "farm/preprocess.h"

#include <algorithm>

namespace farm {

void Preprocessor::FromRgb(const uint8_t* rgb, int w, int h, int row_stride, int rotation_degrees, float* out) {
  const double scale = static_cast<double>(short_side_) / std::min(w, h);
  const int rw = RoundHalfEven(w * scale), rh = RoundHalfEven(h * scale);
  resized_.resize(static_cast<size_t>(rw) * rh * 3);
  resizer_.Resize(rgb, w, h, row_stride, 3, resized_.data(), rw, rh);

  const int left = (rw - size_) / 2, top = (rh - size_) / 2;
  crop_.resize(static_cast<size_t>(size_) * size_ * 3);
  for (int y = 0; y < size_; ++y)
    std::copy_n(&resized_[(static_cast<size_t>(y + top) * rw + left) * 3], static_cast<size_t>(size_) * 3,
                &crop_[static_cast<size_t>(y) * size_ * 3]);

  const int n = size_;
  const int rot = ((rotation_degrees % 360) + 360) % 360;
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      int sx = x, sy = y;  // source pixel for output (x, y) after clockwise rotation
      if (rot == 90) { sx = y; sy = n - 1 - x; }
      else if (rot == 180) { sx = n - 1 - x; sy = n - 1 - y; }
      else if (rot == 270) { sx = n - 1 - y; sy = x; }
      const uint8_t* p = &crop_[(static_cast<size_t>(sy) * n + sx) * 3];
      float* o = out + (static_cast<size_t>(y) * n + x) * 3;
      o[0] = p[0];
      o[1] = p[1];
      o[2] = p[2];
    }
  }
}

void Preprocessor::FromYuv420(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, int y_row_stride,
                              int uv_row_stride, int uv_pixel_stride, int rotation_degrees, float* out) {
  rgb_.resize(static_cast<size_t>(w) * h * 3);
  Yuv420ToRgb(y, u, v, w, h, y_row_stride, uv_row_stride, uv_pixel_stride, rgb_.data());
  FromRgb(rgb_.data(), w, h, w * 3, rotation_degrees, out);
}

}  // namespace farm
