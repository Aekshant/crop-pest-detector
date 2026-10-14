// The optimized resize / box blur / gate metric must be byte-identical to the
// validated reference (tests/ref_quality.h, copied from the Pillow-checked code).
#include <cstdio>
#include <vector>

#include "check.h"
#include "farm/image.h"
#include "farm/quality.h"
#include "ref_quality.h"

namespace {

uint32_t g_seed = 2463534242u;
uint32_t Rand() {  // xorshift32
  g_seed ^= g_seed << 13;
  g_seed ^= g_seed >> 17;
  g_seed ^= g_seed << 5;
  return g_seed;
}

std::vector<uint8_t> Image(int w, int h, int channels, int stride, int kind) {
  std::vector<uint8_t> img(static_cast<size_t>(stride) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w * channels; ++x) {
      const uint32_t r = Rand();
      uint8_t v = static_cast<uint8_t>(r);                                           // noise
      if (kind == 1) v = static_cast<uint8_t>(((x / 7 + y / 5) % 2) * 200 + (r & 31));  // blocks
      if (kind == 2) v = static_cast<uint8_t>((x * 3 + y * 2) & 255);                 // gradient
      if (kind == 3) v = (r & 1) ? 255 : 0;                                            // extremes
      img[static_cast<size_t>(y) * stride + x] = v;
    }
  return img;
}

}  // namespace

void RunFastPathTests() {
  const int sizes[][2] = {{640, 480}, {480, 640}, {256, 256}, {245, 261}, {1000, 750}, {37, 53}, {9, 9}, {1, 1}, {3, 700}};
  const int targets[][2] = {{683, 512}, {512, 683}, {512, 512}, {224, 224}, {256, 341}, {10, 7}, {1, 1}, {700, 3}};
  int cases = 0, mismatches = 0;
  for (auto& s : sizes)
    for (auto& t : targets)
      for (int channels : {1, 3})
        for (int kind = 0; kind < 4; ++kind) {
          const int stride = s[0] * channels + (kind % 2) * 5;  // padded rows too
          auto src = Image(s[0], s[1], channels, stride, kind);
          std::vector<uint8_t> a(static_cast<size_t>(t[0]) * t[1] * channels), b(a.size());
          farm::BilinearResizer resizer;
          resizer.Resize(src.data(), s[0], s[1], stride, channels, a.data(), t[0], t[1]);
          ref::Resize(src.data(), s[0], s[1], stride, channels, b.data(), t[0], t[1]);
          ++cases;
          if (a != b) ++mismatches;
        }
  std::printf("resize: %d cases, %d mismatches\n", cases, mismatches);
  CHECK(mismatches == 0);

  int blur_cases = 0, blur_mismatches = 0;
  for (auto& s : sizes)
    for (int radius : {1, 2, 4, 7})
      for (int kind = 0; kind < 4; ++kind) {
        auto src = Image(s[0], s[1], 1, s[0], kind);
        std::vector<uint8_t> a(src.size()), b(src.size()), tmp_a, tmp_b;
        farm::BoxBlur(src.data(), s[0], s[1], radius, a.data(), &tmp_a);
        ref::BoxBlur(src.data(), s[0], s[1], radius, b.data(), &tmp_b);
        ++blur_cases;
        if (a != b) ++blur_mismatches;
      }
  std::printf("box blur: %d cases, %d mismatches\n", blur_cases, blur_mismatches);
  CHECK(blur_mismatches == 0);

  // The whole gate against the reference pipeline (resize -> blur -> metric).
  int gate_cases = 0, gate_mismatches = 0;
  for (auto& s : sizes)
    for (int kind = 0; kind < 4; ++kind) {
      auto gray = Image(s[0], s[1], 1, s[0] + 3, kind);
      farm::QualityGate gate({});
      const farm::QualityResult r = gate.Check(gray.data(), s[0], s[1], s[0] + 3);
      const double scale = static_cast<double>(farm::QualityGate::kShortSide) / std::min(s[0], s[1]);
      const int sw = std::max(1, farm::RoundHalfEven(s[0] * scale)), sh = std::max(1, farm::RoundHalfEven(s[1] * scale));
      std::vector<uint8_t> small(static_cast<size_t>(sw) * sh), blurred(small.size()), tmp;
      ref::Resize(gray.data(), s[0], s[1], s[0] + 3, 1, small.data(), sw, sh);
      ref::BoxBlur(small.data(), sw, sh, farm::QualityGate::kReblurRadius, blurred.data(), &tmp);
      double blur = 0;
      uint64_t bright = 0;
      ref::Metric(small, blurred, sw, sh, &blur, &bright);
      ++gate_cases;
      if (r.sharpness != static_cast<float>(1.0 - blur / 2.0) ||
          r.brightness != static_cast<float>(static_cast<double>(bright) / small.size())) ++gate_mismatches;
    }
  std::printf("gate: %d cases, %d mismatches\n", gate_cases, gate_mismatches);
  CHECK(gate_mismatches == 0);
}
