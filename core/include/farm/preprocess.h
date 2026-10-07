#pragma once

// Model input preparation, identical to ml/export.py preprocess():
// resize so the short side is 256 (Pillow bilinear), center-crop 224, write
// NHWC float RGB 0..255 straight into the caller's tensor. Camera frames are
// converted from YUV_420_888 first and rotated upright at the end.
// Buffers are reused between frames.

#include <cstdint>
#include <vector>

#include "farm/image.h"

namespace farm {

class Preprocessor {
 public:
  explicit Preprocessor(int input_size = 224, int short_side = 256)
      : size_(input_size), short_side_(short_side) {}

  // rotation_degrees: clockwise rotation that makes the image upright (0/90/180/270).
  void FromRgb(const uint8_t* rgb, int w, int h, int row_stride, int rotation_degrees, float* out);
  void FromYuv420(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, int y_row_stride,
                  int uv_row_stride, int uv_pixel_stride, int rotation_degrees, float* out);

  int input_size() const { return size_; }
  // The 224x224 RGB8 crop from the last call (before rotation), for tests.
  const std::vector<uint8_t>& last_crop() const { return crop_; }

 private:
  int size_, short_side_;
  BilinearResizer resizer_;
  std::vector<uint8_t> rgb_, resized_, crop_;
};

}  // namespace farm
