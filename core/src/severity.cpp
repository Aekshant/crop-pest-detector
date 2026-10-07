#include "farm/severity.h"

#include <algorithm>
#include <cstddef>

namespace farm {
namespace {

// Hue in degrees [0, 360), saturation and value in [0, 1].
void Hsv(int r, int g, int b, float* h, float* s, float* v) {
  const int mx = std::max({r, g, b}), mn = std::min({r, g, b});
  const int d = mx - mn;
  *v = mx / 255.f;
  *s = mx == 0 ? 0.f : static_cast<float>(d) / mx;
  if (d == 0) { *h = 0.f; return; }
  float hh;
  if (mx == r) hh = 60.f * static_cast<float>(g - b) / d;
  else if (mx == g) hh = 60.f * (2.f + static_cast<float>(b - r) / d);
  else hh = 60.f * (4.f + static_cast<float>(r - g) / d);
  *h = hh < 0 ? hh + 360.f : hh;
}

// Binary dilation / erosion with a (2r+1)^2 square, separable.
void Morph(const std::vector<uint8_t>& in, int w, int h, int r, bool dilate, std::vector<uint8_t>* out,
           std::vector<uint8_t>* tmp) {
  tmp->assign(in.size(), 0);
  out->assign(in.size(), 0);
  const uint8_t edge = dilate ? 0 : 1;  // outside the image: background for dilation, foreground for erosion
  auto pass = [&](const std::vector<uint8_t>& src, std::vector<uint8_t>* dst, bool horizontal) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        uint8_t acc = dilate ? 0 : 1;
        for (int k = -r; k <= r; ++k) {
          const int xx = horizontal ? x + k : x, yy = horizontal ? y : y + k;
          const uint8_t v = (xx < 0 || yy < 0 || xx >= w || yy >= h) ? edge : src[static_cast<size_t>(yy) * w + xx];
          acc = dilate ? (acc | v) : (acc & v);
        }
        (*dst)[static_cast<size_t>(y) * w + x] = acc;
      }
  };
  pass(in, tmp, true);
  pass(*tmp, out, false);
}

// Sets holes (background regions not connected to the border) to 1.
void FillHoles(std::vector<uint8_t>* mask, int w, int h, std::vector<uint8_t>* seen) {
  seen->assign(mask->size(), 0);
  std::vector<int> stack;
  auto push = [&](int x, int y) {
    const size_t i = static_cast<size_t>(y) * w + x;
    if (!(*mask)[i] && !(*seen)[i]) { (*seen)[i] = 1; stack.push_back(static_cast<int>(i)); }
  };
  for (int x = 0; x < w; ++x) { push(x, 0); push(x, h - 1); }
  for (int y = 0; y < h; ++y) { push(0, y); push(w - 1, y); }
  while (!stack.empty()) {
    const int i = stack.back();
    stack.pop_back();
    const int x = i % w, y = i / w;
    if (x > 0) push(x - 1, y);
    if (x + 1 < w) push(x + 1, y);
    if (y > 0) push(x, y - 1);
    if (y + 1 < h) push(x, y + 1);
  }
  for (size_t i = 0; i < mask->size(); ++i)
    if (!(*seen)[i]) (*mask)[i] = 1;
}

}  // namespace

SeverityResult SeverityEstimator::Estimate(const uint8_t* rgb, int w, int h, int row_stride) {
  const size_t n = static_cast<size_t>(w) * h;
  green_.assign(n, 0);
  cand_.assign(n, 0);
  strong_.assign(n, 0);
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = rgb + static_cast<size_t>(y) * row_stride;
    for (int x = 0; x < w; ++x) {
      float hu, s, v;
      Hsv(row[x * 3], row[x * 3 + 1], row[x * 3 + 2], &hu, &s, &v);
      const size_t i = static_cast<size_t>(y) * w + x;
      const bool green = hu >= 65.f && hu < 170.f && s >= 0.18f && v >= 0.12f;
      const bool yellow = hu >= 38.f && hu < 65.f && s >= 0.25f && v >= 0.30f;
      const bool brown = (hu < 38.f || hu >= 330.f) && s >= 0.10f && v >= 0.10f && v <= 0.80f;
      const bool dark = v < 0.18f;
      const bool pale = s < 0.15f && v >= 0.35f && v <= 0.85f;
      green_[i] = green;
      cand_[i] = !green && (yellow || brown || dark || pale);
      // Only clearly lesion-coloured pixels may extend the leaf beyond its green
      // tissue: pale and dark also describe paper, sky and shadows.
      strong_[i] = !green && (yellow || brown);
    }
  }

  // 1. Leaf body: green tissue, closed, holes filled.
  Morph(green_, w, h, 2, true, &leaf_, &tmp_);
  Morph(leaf_, w, h, 2, false, &tmp2_, &tmp_);
  leaf_.swap(tmp2_);
  FillHoles(&leaf_, w, h, &tmp_);

  // 2. Grow into clearly lesion-coloured tissue connected to it (necrotic
  //    margins and large dead patches), at most kGrow px from the body.
  constexpr int kGrow = 30;
  std::vector<int>& frontier = queue_;
  frontier.clear();
  dist_.assign(n, -1);
  for (size_t i = 0; i < n; ++i)
    if (leaf_[i]) { dist_[i] = 0; frontier.push_back(static_cast<int>(i)); }
  for (size_t head = 0; head < frontier.size(); ++head) {
    const int i = frontier[head];
    if (dist_[i] >= kGrow) continue;
    const int x = i % w, y = i / w;
    const int nb[4][2] = {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};
    for (const auto& q : nb) {
      if (q[0] < 0 || q[1] < 0 || q[0] >= w || q[1] >= h) continue;
      const int j = q[1] * w + q[0];
      if (dist_[j] >= 0 || !strong_[j]) continue;
      dist_[j] = dist_[i] + 1;
      leaf_[j] = 1;
      frontier.push_back(j);
    }
  }
  // Spots enclosed by the grown leaf (dark or pale centres) are leaf too.
  FillHoles(&leaf_, w, h, &tmp_);

  // 3. Lesions are counted in the leaf interior only: the outline blends leaf
  //    and background into pale/dark pixels that look like lesions.
  Morph(leaf_, w, h, 2, false, &tmp2_, &tmp_);
  lesion_.assign(n, 0);
  size_t leaf = 0, lesion = 0;
  for (size_t i = 0; i < n; ++i) {
    leaf += leaf_[i];
    lesion_[i] = tmp2_[i] && cand_[i];
    lesion += lesion_[i];
  }

  SeverityResult r;
  r.leaf_fraction = static_cast<float>(leaf) / static_cast<float>(n);
  r.lesion_fraction = leaf ? static_cast<float>(lesion) / static_cast<float>(leaf) : 0.f;
  if (r.leaf_fraction < p_.min_leaf_fraction) r.level = SeverityLevel::kUnknown;
  else if (r.lesion_fraction >= p_.high_at) r.level = SeverityLevel::kHigh;
  else if (r.lesion_fraction >= p_.medium_at) r.level = SeverityLevel::kMedium;
  else r.level = SeverityLevel::kLow;
  return r;
}

}  // namespace farm
