// Host micro-benchmark of the quality gate's stages on a 640x480 camera-sized frame.
//   cmake -S core -B core/build-bench -DFARM_CORE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build core/build-bench
//   core/build-bench/quality_bench
#include <chrono>
#include <cstdio>
#include <vector>

#include "farm/image.h"
#include "farm/quality.h"

int main() {
  const int w = 640, h = 480;
  std::vector<uint8_t> gray(static_cast<size_t>(w) * h);
  uint32_t s = 12345;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      s = s * 1664525u + 1013904223u;
      gray[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(((x / 8 + y / 8) % 2) * 120 + (s >> 27) + 40);
    }
  auto ms = [](auto f, int n) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
  };
  const int sw = 683, sh = 512, n = 200;
  farm::BilinearResizer resizer;
  std::vector<uint8_t> small(static_cast<size_t>(sw) * sh), blurred(small.size()), tmp;
  resizer.Resize(gray.data(), w, h, w, 1, small.data(), sw, sh);
  const double t_resize = ms([&] { resizer.Resize(gray.data(), w, h, w, 1, small.data(), sw, sh); }, n);
  const double t_blur = ms([&] { farm::BoxBlur(small.data(), sw, sh, 4, blurred.data(), &tmp); }, n);
  farm::QualityGate gate({});
  farm::QualityResult r;
  const double t_total = ms([&] { r = gate.Check(gray.data(), w, h, w); }, n);
  std::printf("resize %.3f ms  blur %.3f ms  diff (rest) %.3f ms  total %.3f ms   sharpness %.9f brightness %.6f\n",
              t_resize, t_blur, t_total - t_resize - t_blur, t_total, r.sharpness, r.brightness);
}
