#pragma once

// Vision engine: quality gate -> preprocess -> LiteRT inference -> decision.
// Create once (with a warm-up run), call from ONE thread only, release when the
// app is backgrounded (design doc: interpreter and GPU delegate live on a single
// dedicated thread and are never shared).
//
// Delegate order: GPU delegate with the FP16 model, else CPU (XNNPACK) with the
// INT8 model. Each model file carries its own temperature and reject threshold.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "farm/decision.h"
#include "farm/preprocess.h"
#include "farm/quality.h"
#include "farm/severity.h"

namespace farm {

constexpr int kMaxLabels = 32;

struct ModelFile {
  std::vector<uint8_t> bytes;  // .tflite contents (kept alive by the engine)
  float temperature = 1.f;
  float reject_below = 0.5f;
};

struct EngineConfig {
  ModelFile gpu_model;   // FP16
  ModelFile cpu_model;   // INT8
  float confident_at = 0.8f;
  int unknown_index = 0;
  int num_labels = 12;
  int input_size = 224;
  int num_threads = 4;
  bool allow_gpu = true;
  QualityThresholds quality;
  // 1 for labels that get a lesion-area severity estimate (diseases only).
  std::vector<uint8_t> severity_labels;
};

enum class Backend : int { kGpu = 0, kCpu = 1 };

struct VisionResult {
  QualityResult quality;
  Decision decision = Decision::kRetake;
  int top = -1;
  float top_prob = 0.f;
  std::array<float, kMaxLabels> probs{};
  SeverityResult severity;  // level kUnknown unless a disease label was given
  float quality_ms = 0.f, preprocess_ms = 0.f, inference_ms = 0.f, severity_ms = 0.f;
};

class Engine {
 public:
  // Returns nullptr and sets *error when no backend can be created.
  static std::unique_ptr<Engine> Create(EngineConfig config, std::string* error);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  Backend backend() const { return backend_; }
  float warmup_ms() const { return warmup_ms_; }
  // Why the GPU delegate was not used (empty when it was).
  const std::string& gpu_fallback_reason() const { return gpu_fallback_reason_; }

  QualityResult CheckQuality(const uint8_t* gray, int w, int h, int row_stride);

  VisionResult AnalyzeYuv(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, int y_row_stride,
                          int uv_row_stride, int uv_pixel_stride, int rotation_degrees);
  VisionResult AnalyzeRgb(const uint8_t* rgb, int w, int h, int row_stride, int rotation_degrees);

 private:
  struct Impl;
  explicit Engine(EngineConfig config);
  VisionResult Infer(VisionResult r);

  EngineConfig config_;
  std::unique_ptr<Impl> impl_;
  Backend backend_ = Backend::kCpu;
  DecisionThresholds decision_;
  float warmup_ms_ = 0.f;
  std::string gpu_fallback_reason_;
  QualityGate gate_;
  Preprocessor pre_;
  SeverityEstimator severity_;
};

}  // namespace farm
