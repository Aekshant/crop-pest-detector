#include "farm/quality.h"

#include <algorithm>
#include <cstdlib>

namespace farm {

QualityResult QualityGate::Check(const uint8_t* gray, int w, int h, int row_stride) {
  const double scale = static_cast<double>(kShortSide) / std::min(w, h);
  const int sw = std::max(1, RoundHalfEven(w * scale));
  const int sh = std::max(1, RoundHalfEven(h * scale));
  small_.resize(static_cast<size_t>(sw) * sh);
  blurred_.resize(small_.size());
  resizer_.Resize(gray, w, h, row_stride, 1, small_.data(), sw, sh);
  BoxBlur(small_.data(), sw, sh, kReblurRadius, blurred_.data(), &tmp_);

  // Neighbour differences with wrap-around (PIL ImageChops.offset), original
  // vs re-blurred; the share of edge energy the re-blur removes is sharpness.
  double blur = 0.0;
  uint64_t brightness_sum = 0;
  for (int dir = 0; dir < 2; ++dir) {
    uint64_t s_orig = 0, s_lost = 0;
    for (int y = 0; y < sh; ++y) {
      const int py = dir == 1 ? (y + sh - 1) % sh : y;
      const uint8_t* g = &small_[static_cast<size_t>(y) * sw];
      const uint8_t* gp = &small_[static_cast<size_t>(py) * sw];
      const uint8_t* b = &blurred_[static_cast<size_t>(y) * sw];
      const uint8_t* bp = &blurred_[static_cast<size_t>(py) * sw];
      for (int x = 0; x < sw; ++x) {
        const int px = dir == 0 ? (x + sw - 1) % sw : x;
        const int d_orig = std::abs(g[x] - gp[px]);
        const int d_blur = std::abs(b[x] - bp[px]);
        s_orig += d_orig;
        s_lost += std::max(0, d_orig - d_blur);
        if (dir == 0) brightness_sum += g[x];
      }
    }
    blur += static_cast<double>(s_orig - s_lost) / static_cast<double>(std::max<uint64_t>(s_orig, 1));
  }

  QualityResult r;
  r.sharpness = static_cast<float>(1.0 - blur / 2.0);
  r.brightness = static_cast<float>(static_cast<double>(brightness_sum) / small_.size());
  if (r.sharpness < t_.min_sharpness) r.verdict = QualityVerdict::kBlurry;
  else if (r.brightness < t_.min_brightness) r.verdict = QualityVerdict::kTooDark;
  else if (r.brightness > t_.max_brightness) r.verdict = QualityVerdict::kTooBright;
  return r;
}

QualityResult QualityGate::CheckRgb(const uint8_t* rgb, int w, int h, int row_stride) {
  luma_.resize(static_cast<size_t>(w) * h);
  RgbToLuma(rgb, w, h, row_stride, luma_.data());
  return Check(luma_.data(), w, h, w);
}

}  // namespace farm
