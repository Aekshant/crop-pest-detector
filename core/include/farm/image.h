#pragma once

// Image primitives that reproduce the Python reference (Pillow) exactly, so the
// device preprocesses and judges photos the way the model was trained/tuned:
//   - BilinearResizer: PIL Image.resize(..., BILINEAR) (antialiased triangle
//     filter, 22-bit fixed point, horizontal pass then vertical, uint8 between)
//   - BoxBlur: PIL ImageFilter.BoxBlur(radius) for integer radius
//   - RgbToLuma: PIL convert("L") (ITU-R 601-2 luma, 16-bit fixed point)
// All functions write into caller-owned buffers; no per-frame allocation once
// buffers have their final size.

#include <cstdint>
#include <vector>

namespace farm {

// Resizes interleaved 8-bit images with `channels` channels (1 or 3).
class BilinearResizer {
 public:
  void Resize(const uint8_t* src, int src_w, int src_h, int src_stride, int channels,
              uint8_t* dst, int dst_w, int dst_h);

 private:
  struct Coeffs {
    int in = 0, out = 0, ksize = 0;
    std::vector<int> bounds;   // per output: first input index, count
    std::vector<int32_t> k;    // out * ksize fixed-point weights
  };
  static void Compute(int in, int out, Coeffs* c);
  Coeffs h_, v_;
  std::vector<uint8_t> tmp_;  // horizontal pass output
};

// PIL BoxBlur with integer radius on a single-channel image (edges replicated).
void BoxBlur(const uint8_t* src, int w, int h, int radius, uint8_t* dst, std::vector<uint8_t>* tmp);

// PIL convert("L") from interleaved RGB8.
void RgbToLuma(const uint8_t* rgb, int w, int h, int stride, uint8_t* dst);

// Android YUV_420_888 (full-range BT.601, as produced by CameraX) to RGB8.
void Yuv420ToRgb(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h,
                 int y_row_stride, int uv_row_stride, int uv_pixel_stride, uint8_t* rgb);

// Python round(): half to even.
int RoundHalfEven(double x);

}  // namespace farm
