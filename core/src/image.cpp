#include "farm/image.h"

#include <algorithm>
#include <cmath>

namespace farm {
namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;  // Pillow's PRECISION_BITS for 8-bit images

inline uint8_t Clip8(int64_t v) {
  if (v >= (int64_t{1} << kPrecisionBits << 8)) return 255;
  if (v <= 0) return 0;
  return static_cast<uint8_t>(v >> kPrecisionBits);
}

inline uint8_t Clip8(int32_t v) {
  if (v >= (1 << kPrecisionBits << 8)) return 255;
  if (v <= 0) return 0;
  return static_cast<uint8_t>(v >> kPrecisionBits);
}

inline double Triangle(double x) {
  x = std::fabs(x);
  return x < 1.0 ? 1.0 - x : 0.0;
}

}  // namespace

int RoundHalfEven(double x) { return static_cast<int>(std::nearbyint(x)); }

// Mirrors Pillow's precompute_coeffs + normalize_coeffs_8bpc (Resample.c).
void BilinearResizer::Compute(int in, int out, Coeffs* c) {
  if (c->in == in && c->out == out) return;
  const double scale = static_cast<double>(in) / out;
  const double filterscale = std::max(scale, 1.0);
  const double support = 1.0 * filterscale;
  const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  std::vector<double> pre(static_cast<size_t>(out) * ksize, 0.0);
  c->bounds.assign(static_cast<size_t>(out) * 2, 0);
  for (int xx = 0; xx < out; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in) xmax = in;
    xmax -= xmin;
    double* k = &pre[static_cast<size_t>(xx) * ksize];
    double ww = 0.0;
    for (int x = 0; x < xmax; ++x) {
      const double w = Triangle((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x) {
      if (ww != 0.0) k[x] /= ww;
    }
    c->bounds[xx * 2] = xmin;
    c->bounds[xx * 2 + 1] = xmax;
  }
  c->k.resize(pre.size());
  for (size_t i = 0; i < pre.size(); ++i) {
    const double v = pre[i] * (1 << kPrecisionBits);
    c->k[i] = static_cast<int32_t>(pre[i] < 0 ? -0.5 + v : 0.5 + v);
  }
  c->in = in;
  c->out = out;
  c->ksize = ksize;
}

void BilinearResizer::Resize(const uint8_t* src, int src_w, int src_h, int src_stride, int channels,
                             uint8_t* dst, int dst_w, int dst_h) {
  const bool need_h = dst_w != src_w;
  const bool need_v = dst_h != src_h;
  Compute(src_w, dst_w, &h_);
  Compute(src_h, dst_h, &v_);

  // Rows of the source the vertical pass will read (Pillow limits the
  // horizontal pass to them; results are identical).
  int y_first = 0, y_last = src_h;
  if (need_v) {
    y_first = v_.bounds[0];
    y_last = v_.bounds[(dst_h - 1) * 2] + v_.bounds[(dst_h - 1) * 2 + 1];
  }

  const uint8_t* hsrc = src;
  int hstride = src_stride;
  int hrows = src_h;
  if (need_h) {
    hrows = y_last - y_first;
    tmp_.resize(static_cast<size_t>(hrows) * dst_w * channels);
    for (int y = 0; y < hrows; ++y) {
      const uint8_t* row = src + static_cast<size_t>(y + y_first) * src_stride;
      uint8_t* out = &tmp_[static_cast<size_t>(y) * dst_w * channels];
      // 32-bit accumulators, as Pillow's 8bpc path: 255 * 2^22 * (sum of weights ~ 1) + 2^21 < 2^31.
      if (channels == 1) {
        for (int xx = 0; xx < dst_w; ++xx) {
          const int xmin = h_.bounds[xx * 2], xmax = h_.bounds[xx * 2 + 1];
          const int32_t* k = &h_.k[static_cast<size_t>(xx) * h_.ksize];
          const uint8_t* p = row + xmin;
          int32_t ss = 1 << (kPrecisionBits - 1);
          for (int x = 0; x < xmax; ++x) ss += p[x] * k[x];
          out[xx] = Clip8(ss);
        }
      } else {
        for (int xx = 0; xx < dst_w; ++xx) {
          const int xmin = h_.bounds[xx * 2], xmax = h_.bounds[xx * 2 + 1];
          const int32_t* k = &h_.k[static_cast<size_t>(xx) * h_.ksize];
          for (int c = 0; c < channels; ++c) {
            int32_t ss = 1 << (kPrecisionBits - 1);
            for (int x = 0; x < xmax; ++x) ss += row[(x + xmin) * channels + c] * k[x];
            out[xx * channels + c] = Clip8(ss);
          }
        }
      }
    }
    hsrc = tmp_.data();
    hstride = dst_w * channels;
  } else {
    hsrc = src + static_cast<size_t>(y_first) * src_stride;
  }

  const int width = need_h ? dst_w : src_w;
  if (!need_v) {
    for (int y = 0; y < dst_h; ++y)
      std::copy_n(hsrc + static_cast<size_t>(y) * hstride, static_cast<size_t>(width) * channels,
                  dst + static_cast<size_t>(y) * width * channels);
    return;
  }
  // Row by row (not column by column): the same sums, read in memory order.
  const int n = width * channels;
  acc_.resize(static_cast<size_t>(n));
  for (int yy = 0; yy < dst_h; ++yy) {
    const int ymin = v_.bounds[yy * 2] - y_first, ymax = v_.bounds[yy * 2 + 1];
    const int32_t* k = &v_.k[static_cast<size_t>(yy) * v_.ksize];
    int32_t* acc = acc_.data();
    std::fill_n(acc, n, 1 << (kPrecisionBits - 1));
    for (int y = 0; y < ymax; ++y) {
      const uint8_t* row = hsrc + static_cast<size_t>(y + ymin) * hstride;
      const int32_t ky = k[y];
      for (int x = 0; x < n; ++x) acc[x] += row[x] * ky;
    }
    uint8_t* out = dst + static_cast<size_t>(yy) * n;
    for (int x = 0; x < n; ++x) out[x] = Clip8(acc[x]);
  }
}

namespace {

// Pillow's ImagingLineBoxBlur8 for an integer radius: clamped (edge-replicated)
// window sum, then a 24-bit fixed-point mean (fractional-radius term is 0).
inline uint8_t BoxMean(uint32_t acc, uint32_t ww) {
  return static_cast<uint8_t>((static_cast<uint64_t>(acc) * ww + (1u << 23)) >> 24);
}

}  // namespace

void BoxBlur(const uint8_t* src, int w, int h, int radius, uint8_t* dst, std::vector<uint8_t>* tmp) {
  const uint32_t ww = static_cast<uint32_t>(static_cast<float>(1 << 24) / (radius * 2 + 1));
  tmp->resize(static_cast<size_t>(w) * h);
  // Horizontal: one sliding window per row.
  for (int y = 0; y < h; ++y) {
    const uint8_t* in = src + static_cast<size_t>(y) * w;
    uint8_t* out = tmp->data() + static_cast<size_t>(y) * w;
    uint32_t acc = 0;
    for (int i = -radius; i <= radius; ++i) acc += in[std::clamp(i, 0, w - 1)];
    for (int x = 0; x < w; ++x) {
      out[x] = BoxMean(acc, ww);
      acc += in[std::min(x + radius + 1, w - 1)] - in[std::max(x - radius, 0)];
    }
  }
  // Vertical: a row of per-column windows slid down the image (memory order, vectorizable).
  std::vector<uint32_t> acc(static_cast<size_t>(w), 0);
  const uint8_t* t = tmp->data();
  for (int i = -radius; i <= radius; ++i) {
    const uint8_t* row = t + static_cast<size_t>(std::clamp(i, 0, h - 1)) * w;
    for (int x = 0; x < w; ++x) acc[x] += row[x];
  }
  for (int y = 0; y < h; ++y) {
    uint8_t* out = dst + static_cast<size_t>(y) * w;
    for (int x = 0; x < w; ++x) out[x] = BoxMean(acc[x], ww);
    const uint8_t* add = t + static_cast<size_t>(std::min(y + radius + 1, h - 1)) * w;
    const uint8_t* sub = t + static_cast<size_t>(std::max(y - radius, 0)) * w;
    for (int x = 0; x < w; ++x) acc[x] += add[x] - sub[x];
  }
}

void RgbToLuma(const uint8_t* rgb, int w, int h, int stride, uint8_t* dst) {
  for (int y = 0; y < h; ++y) {
    const uint8_t* p = rgb + static_cast<size_t>(y) * stride;
    uint8_t* o = dst + static_cast<size_t>(y) * w;
    for (int x = 0; x < w; ++x, p += 3)
      o[x] = static_cast<uint8_t>((p[0] * 19595u + p[1] * 38470u + p[2] * 7471u + 0x8000u) >> 16);
  }
}

void Yuv420ToRgb(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h,
                 int y_row_stride, int uv_row_stride, int uv_pixel_stride, uint8_t* rgb) {
  // Full-range BT.601 (JFIF), 16-bit fixed point.
  constexpr int kRv = 91881, kGu = 22554, kGv = 46802, kBu = 116130;
  for (int j = 0; j < h; ++j) {
    const uint8_t* yr = y + static_cast<size_t>(j) * y_row_stride;
    const uint8_t* ur = u + static_cast<size_t>(j / 2) * uv_row_stride;
    const uint8_t* vr = v + static_cast<size_t>(j / 2) * uv_row_stride;
    uint8_t* o = rgb + static_cast<size_t>(j) * w * 3;
    for (int i = 0; i < w; ++i) {
      const int yy = yr[i] << 16;
      const int uu = ur[(i / 2) * uv_pixel_stride] - 128;
      const int vv = vr[(i / 2) * uv_pixel_stride] - 128;
      o[i * 3 + 0] = static_cast<uint8_t>(std::clamp((yy + kRv * vv + 0x8000) >> 16, 0, 255));
      o[i * 3 + 1] = static_cast<uint8_t>(std::clamp((yy - kGu * uu - kGv * vv + 0x8000) >> 16, 0, 255));
      o[i * 3 + 2] = static_cast<uint8_t>(std::clamp((yy + kBu * uu + 0x8000) >> 16, 0, 255));
    }
  }
}

}  // namespace farm
