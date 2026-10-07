#pragma once

// Lesion-area severity estimate (design doc, vision step 7): for disease labels
// only, the share of leaf area that is lesion, from HSV colour segmentation.
// Always shown as "estimated"; never derived from model confidence.
//
// Leaf    = green tissue, closed and hole-filled (lesions inside the leaf count
//           as leaf), grown up to 20 px into connected yellow/brown tissue
//           (necrotic margins, dead patches), holes filled again.
// Lesion  = leaf-interior pixels (2 px in from the outline) that are yellow
//           (chlorosis), brown (necrosis), very dark (spots) or pale grey
//           (mould) instead of green.
// Meant for a close-up of one leaf; on whole-plant scenes it is rough.

#include <cstdint>
#include <vector>

namespace farm {

enum class SeverityLevel : int { kUnknown = 0, kLow = 1, kMedium = 2, kHigh = 3 };

struct SeverityResult {
  float lesion_fraction = 0.f;  // lesion pixels / leaf pixels
  float leaf_fraction = 0.f;    // leaf pixels / image pixels
  SeverityLevel level = SeverityLevel::kUnknown;
};

struct SeverityParams {
  float min_leaf_fraction = 0.15f;  // less leaf than this: can't estimate
  float medium_at = 0.10f;
  float high_at = 0.30f;
};

class SeverityEstimator {
 public:
  explicit SeverityEstimator(SeverityParams p = {}) : p_(p) {}

  // rgb: interleaved RGB8 (the model's 224x224 crop).
  SeverityResult Estimate(const uint8_t* rgb, int w, int h, int row_stride);

  // Masks from the last call (1 = set), for tests and visualisation.
  const std::vector<uint8_t>& leaf_mask() const { return leaf_; }
  const std::vector<uint8_t>& lesion_mask() const { return lesion_; }

 private:
  SeverityParams p_;
  std::vector<uint8_t> green_, cand_, strong_, leaf_, lesion_, tmp_, tmp2_;
  std::vector<int> dist_, queue_;
};

}  // namespace farm
