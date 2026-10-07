#pragma once

// Quality gate (design doc, vision step 2): runs before any model.
// Same metric as ml/quality_gate.py: Crete-Roffet "blur effect" on the gray
// image resized to a 512 px short side (Pillow-exact resize and box blur),
// plus mean brightness. Thresholds come from the model manifest.

#include <cstdint>
#include <vector>

#include "farm/image.h"

namespace farm {

struct QualityThresholds {
  float min_sharpness = 0.f;
  float min_brightness = 0.f;
  float max_brightness = 255.f;
};

enum class QualityVerdict : int { kOk = 0, kBlurry = 1, kTooDark = 2, kTooBright = 3 };

struct QualityResult {
  float sharpness = 0.f;   // 1 - blur effect: 0 blurry .. 1 sharp
  float brightness = 0.f;  // mean gray 0..255
  QualityVerdict verdict = QualityVerdict::kOk;
};

class QualityGate {
 public:
  static constexpr int kShortSide = 512;
  static constexpr int kReblurRadius = 4;

  explicit QualityGate(QualityThresholds t) : t_(t) {}

  // gray: 8-bit luma (camera Y plane or RgbToLuma output).
  QualityResult Check(const uint8_t* gray, int w, int h, int row_stride);
  // Convenience for RGB8 input (golden tests): PIL convert("L") first.
  QualityResult CheckRgb(const uint8_t* rgb, int w, int h, int row_stride);

  const QualityThresholds& thresholds() const { return t_; }

 private:
  QualityThresholds t_;
  BilinearResizer resizer_;
  std::vector<uint8_t> luma_, small_, blurred_, tmp_;
};

}  // namespace farm
