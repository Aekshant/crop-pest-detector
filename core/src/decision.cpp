#include "farm/decision.h"

#include <algorithm>
#include <cmath>

namespace farm {

Decision Decide(const float* logits, size_t n, const DecisionThresholds& t, float* probs, int* top) {
  const float m = *std::max_element(logits, logits + n);
  double z = 0.0;
  for (size_t i = 0; i < n; ++i) z += probs[i] = std::exp((logits[i] - m) / t.temperature);
  for (size_t i = 0; i < n; ++i) probs[i] = static_cast<float>(probs[i] / z);
  *top = static_cast<int>(std::max_element(probs, probs + n) - probs);
  const float p = probs[*top];
  if (*top == t.unknown_index || p < t.reject_below) return Decision::kNotRecognised;
  if (p < t.confident_at) return Decision::kUncertain;
  return Decision::kConfident;
}

}  // namespace farm
