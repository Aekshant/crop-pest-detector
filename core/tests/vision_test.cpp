// Host tests for the vision core: Python parity on golden images (A4, A5),
// YUV path, rotation, decision rules, and quality-gate speed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "farm/decision.h"
#include "farm/image.h"
#include "farm/preprocess.h"
#include "farm/quality.h"
#include "farm/severity.h"

namespace {

std::vector<uint8_t> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Golden {
  std::string name;
  int w, h, label;
  double sharpness, brightness;
};

std::vector<Golden> ReadGolden() {
  std::ifstream f(std::string(FARM_GOLDEN_DIR) + "/golden.tsv");
  std::string line;
  std::getline(f, line);  // header
  std::vector<Golden> out;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    Golden g;
    s >> g.name >> g.w >> g.h >> g.label >> g.sharpness >> g.brightness;
    if (!g.name.empty()) out.push_back(g);
  }
  return out;
}

// RGB -> YUV_420_888-like planes (full-range BT.601, 2x2 averaged chroma), the
// inverse of what CameraX hands us; used to test the YUV path.
void RgbToYuv420(const std::vector<uint8_t>& rgb, int w, int h, std::vector<uint8_t>* y, std::vector<uint8_t>* u,
                 std::vector<uint8_t>* v) {
  y->resize(static_cast<size_t>(w) * h);
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  u->assign(static_cast<size_t>(cw) * ch, 0);
  v->assign(u->size(), 0);
  std::vector<double> su(u->size(), 0), sv(u->size(), 0), n(u->size(), 0);
  for (int j = 0; j < h; ++j)
    for (int i = 0; i < w; ++i) {
      const uint8_t* p = &rgb[(static_cast<size_t>(j) * w + i) * 3];
      const double r = p[0], g = p[1], b = p[2];
      (*y)[static_cast<size_t>(j) * w + i] = static_cast<uint8_t>(std::clamp(std::lround(0.299 * r + 0.587 * g + 0.114 * b), 0L, 255L));
      const size_t c = static_cast<size_t>(j / 2) * cw + i / 2;
      su[c] += -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
      sv[c] += 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
      n[c] += 1;
    }
  for (size_t c = 0; c < u->size(); ++c) {
    (*u)[c] = static_cast<uint8_t>(std::clamp(std::lround(su[c] / n[c]), 0L, 255L));
    (*v)[c] = static_cast<uint8_t>(std::clamp(std::lround(sv[c] / n[c]), 0L, 255L));
  }
}

void TestGoldenParity() {
  const auto golden = ReadGolden();
  CHECK(golden.size() >= 5);
  farm::Preprocessor pre;
  farm::QualityGate gate({0.5f, 40.f, 240.f});
  std::vector<float> tensor(224 * 224 * 3), tensor_yuv(tensor.size());
  for (const auto& g : golden) {
    const auto rgb = ReadFile(std::string(FARM_GOLDEN_DIR) + "/" + g.name + ".rgb");
    const auto ref = ReadFile(std::string(FARM_GOLDEN_DIR) + "/" + g.name + ".pre");
    CHECK(rgb.size() == static_cast<size_t>(g.w) * g.h * 3);
    CHECK(ref.size() == tensor.size());
    if (rgb.size() != static_cast<size_t>(g.w) * g.h * 3 || ref.size() != tensor.size()) continue;

    // A5: preprocessing must match Python exactly.
    pre.FromRgb(rgb.data(), g.w, g.h, g.w * 3, 0, tensor.data());
    int max_diff = 0, diffs = 0;
    for (size_t i = 0; i < tensor.size(); ++i) {
      const int d = std::abs(static_cast<int>(tensor[i]) - ref[i]);
      max_diff = std::max(max_diff, d);
      diffs += d != 0;
    }
    CHECK(max_diff == 0);

    // A4: quality metric must match Python.
    const auto q = gate.CheckRgb(rgb.data(), g.w, g.h, g.w * 3);
    CHECK(std::fabs(q.sharpness - g.sharpness) < 1e-5);
    CHECK(std::fabs(q.brightness - g.brightness) < 1e-3);

    // Camera path: same photo as YUV 4:2:0 must give nearly the same tensor
    // (only chroma subsampling differs).
    std::vector<uint8_t> y, u, v;
    RgbToYuv420(rgb, g.w, g.h, &y, &u, &v);
    pre.FromYuv420(y.data(), u.data(), v.data(), g.w, g.h, g.w, (g.w + 1) / 2, 1, 0, tensor_yuv.data());
    double mad = 0;
    for (size_t i = 0; i < tensor.size(); ++i) mad += std::fabs(tensor[i] - tensor_yuv[i]);
    mad /= tensor.size();
    CHECK(mad < 4.0);
    std::printf("  %-26s preprocess max|diff| %d (%d px)  sharpness %.6f vs %.6f  brightness %.3f vs %.3f  yuv MAD %.2f\n",
                g.name.c_str(), max_diff, diffs, q.sharpness, g.sharpness, q.brightness, g.brightness, mad);
  }
}

void TestRotation() {
  // 224x224 crop of a 224x224 image is the image itself; check one marked pixel.
  std::vector<uint8_t> rgb(224 * 224 * 3, 0);
  rgb[(0 * 224 + 10) * 3] = 255;  // red pixel at (x=10, y=0)
  farm::Preprocessor pre(224, 224);
  std::vector<float> out(224 * 224 * 3);
  pre.FromRgb(rgb.data(), 224, 224, 224 * 3, 90, out.data());
  // Rotating 90 deg clockwise moves (x=10, y=0) to (x=223, y=10).
  CHECK(out[(10 * 224 + 223) * 3] == 255.f);
  pre.FromRgb(rgb.data(), 224, 224, 224 * 3, 180, out.data());
  CHECK(out[(223 * 224 + 213) * 3] == 255.f);
  pre.FromRgb(rgb.data(), 224, 224, 224 * 3, 270, out.data());
  CHECK(out[(213 * 224 + 0) * 3] == 255.f);
}

void TestDecision() {
  farm::DecisionThresholds t{1.0f, 0.35f, 0.8f, 0};
  float probs[3];
  int top = -1;
  const float confident[3] = {0.f, 5.f, 0.f};
  CHECK(farm::Decide(confident, 3, t, probs, &top) == farm::Decision::kConfident && top == 1);
  CHECK(std::fabs(probs[0] + probs[1] + probs[2] - 1.f) < 1e-5);
  const float uncertain[3] = {0.f, 1.f, 0.f};  // p = 0.576
  CHECK(farm::Decide(uncertain, 3, t, probs, &top) == farm::Decision::kUncertain);
  float probs4[4];
  const float flat[4] = {0.f, 0.1f, 0.f, 0.f};  // top p = 0.27 < 0.35
  CHECK(farm::Decide(flat, 4, t, probs4, &top) == farm::Decision::kNotRecognised && top == 1);
  const float unknown[3] = {5.f, 0.f, 0.f};
  CHECK(farm::Decide(unknown, 3, t, probs, &top) == farm::Decision::kNotRecognised && top == 0);
  t.temperature = 4.f;  // a hotter temperature flattens the same logits
  CHECK(farm::Decide(confident, 3, t, probs, &top) == farm::Decision::kUncertain);
}

void TestQualitySpeed() {
  // A4 budget: < 10 ms per frame on the analysis frame size (640x480 Y plane).
  std::vector<uint8_t> y(640 * 480);
  for (size_t i = 0; i < y.size(); ++i) y[i] = static_cast<uint8_t>((i * 7919) % 251);
  farm::QualityGate gate({0.5f, 40.f, 240.f});
  gate.Check(y.data(), 640, 480, 640);  // size buffers once
  const auto t = std::chrono::steady_clock::now();
  const int n = 20;
  for (int i = 0; i < n; ++i) gate.Check(y.data(), 640, 480, 640);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count() / n;
  std::printf("  quality gate 640x480: %.2f ms/frame (host)\n", ms);
  CHECK(ms < 10.0);
}

void TestSeverity() {
  // 224x224: white background, green leaf disc (r=90), brown spots inside it.
  const int n = 224;
  std::vector<uint8_t> rgb(n * n * 3, 255);
  int leaf_px = 0, spot_px = 0;
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x) {
      const int dx = x - 112, dy = y - 112;
      uint8_t* p = &rgb[(static_cast<size_t>(y) * n + x) * 3];
      if (dx * dx + dy * dy > 90 * 90) continue;
      ++leaf_px;
      const bool spot = ((x / 20) % 3 == 0) && ((y / 20) % 3 == 0) && dx * dx + dy * dy < 80 * 80;
      if (spot) { p[0] = 120; p[1] = 70; p[2] = 30; ++spot_px; }   // brown
      else { p[0] = 60; p[1] = 140; p[2] = 50; }                   // green
    }
  farm::SeverityEstimator est;
  const auto r = est.Estimate(rgb.data(), n, n, n * 3);
  const float truth = static_cast<float>(spot_px) / leaf_px;
  std::printf("  severity synthetic: estimated %.3f vs truth %.3f, leaf %.3f\n", r.lesion_fraction, truth,
              r.leaf_fraction);
  CHECK(std::fabs(r.lesion_fraction - truth) < 0.02f);
  CHECK(std::fabs(r.leaf_fraction - static_cast<float>(leaf_px) / (n * n)) < 0.02f);

  // Healthy leaf: no lesions, low severity.
  for (size_t i = 0; i < rgb.size(); i += 3)
    if (rgb[i] == 120) { rgb[i] = 60; rgb[i + 1] = 140; rgb[i + 2] = 50; }
  const auto healthy = est.Estimate(rgb.data(), n, n, n * 3);
  CHECK(healthy.lesion_fraction < 0.01f && healthy.level == farm::SeverityLevel::kLow);

  // No leaf (white wall): can't estimate.
  std::vector<uint8_t> wall(n * n * 3, 250);
  CHECK(est.Estimate(wall.data(), n, n, n * 3).level == farm::SeverityLevel::kUnknown);
}

}  // namespace

void RunVisionTests() {
  TestSeverity();
  TestGoldenParity();
  TestRotation();
  TestDecision();
  TestQualitySpeed();
}
