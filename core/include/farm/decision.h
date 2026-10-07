#pragma once

// Turns model logits into what the farmer sees, using the manifest's per-file
// temperature and thresholds (design doc, vision step 6).

#include <cstddef>

namespace farm {

enum class Decision : int { kRetake = 0, kNotRecognised = 1, kUncertain = 2, kConfident = 3 };

struct DecisionThresholds {
  float temperature = 1.f;
  float reject_below = 0.5f;
  float confident_at = 0.8f;
  int unknown_index = 0;
};

// probs = softmax(logits / temperature); returns the top index and decision.
Decision Decide(const float* logits, size_t n, const DecisionThresholds& t, float* probs, int* top);

}  // namespace farm
