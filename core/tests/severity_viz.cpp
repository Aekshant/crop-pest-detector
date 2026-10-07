// severity_viz in.rgb w h out.rgb : writes an overlay (lesion red, leaf kept,
// background darkened) and prints "lesion_fraction leaf_fraction level".
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "farm/severity.h"

int main(int argc, char** argv) {
  if (argc != 5) { std::fprintf(stderr, "usage: severity_viz in.rgb w h out.rgb\n"); return 2; }
  const int w = std::atoi(argv[2]), h = std::atoi(argv[3]);
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<uint8_t> rgb((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (rgb.size() != static_cast<size_t>(w) * h * 3) { std::fprintf(stderr, "bad size\n"); return 2; }
  farm::SeverityEstimator est;
  const auto r = est.Estimate(rgb.data(), w, h, w * 3);
  for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
    uint8_t* p = &rgb[i * 3];
    if (est.lesion_mask()[i]) { p[0] = 255; p[1] = 0; p[2] = 0; }
    else if (!est.leaf_mask()[i]) { p[0] /= 4; p[1] /= 4; p[2] /= 4; }
  }
  std::ofstream(argv[4], std::ios::binary).write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  std::printf("%.4f %.4f %d\n", r.lesion_fraction, r.leaf_fraction, static_cast<int>(r.level));
  return 0;
}
