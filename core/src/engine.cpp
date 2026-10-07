#include "farm/engine.h"

#include <chrono>
#include <cstring>

#include "tflite/c/c_api.h"
#include "tflite/delegates/gpu/delegate.h"

namespace farm {
namespace {

using Clock = std::chrono::steady_clock;
float MsSince(Clock::time_point t) {
  return std::chrono::duration<float, std::milli>(Clock::now() - t).count();
}

}  // namespace

struct Engine::Impl {
  TfLiteModel* model = nullptr;
  TfLiteInterpreterOptions* options = nullptr;
  TfLiteInterpreter* interpreter = nullptr;
  TfLiteDelegate* gpu = nullptr;

  ~Impl() { Reset(); }
  void Reset() {
    if (interpreter) TfLiteInterpreterDelete(interpreter);
    if (options) TfLiteInterpreterOptionsDelete(options);
    if (gpu) TfLiteGpuDelegateV2Delete(gpu);
    if (model) TfLiteModelDelete(model);
    interpreter = nullptr;
    options = nullptr;
    gpu = nullptr;
    model = nullptr;
  }

  // Builds an interpreter for `bytes`; with use_gpu the GPU delegate is applied.
  bool Build(const std::vector<uint8_t>& bytes, bool use_gpu, int threads, std::string* why) {
    Reset();
    model = TfLiteModelCreate(bytes.data(), bytes.size());
    if (!model) { *why = "model failed to load"; return false; }
    options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, threads);
    if (use_gpu) {
      TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
      o.is_precision_loss_allowed = 1;  // FP16 compute
      o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
      o.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      gpu = TfLiteGpuDelegateV2Create(&o);
      if (!gpu) { *why = "GPU delegate unavailable"; return false; }
      TfLiteInterpreterOptionsAddDelegate(options, gpu);
    }
    interpreter = TfLiteInterpreterCreate(model, options);
    if (!interpreter) { *why = use_gpu ? "GPU delegate rejected the model" : "interpreter creation failed"; return false; }
    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) { *why = "tensor allocation failed"; return false; }
    return true;
  }
};

Engine::Engine(EngineConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()), gate_(config_.quality),
      pre_(config_.input_size) {}

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::Create(EngineConfig config, std::string* error) {
  if (config.num_labels > kMaxLabels) { *error = "too many labels"; return nullptr; }
  std::unique_ptr<Engine> e(new Engine(std::move(config)));
  std::string why;
  bool ok = false;
  if (e->config_.allow_gpu && !e->config_.gpu_model.bytes.empty()) {
    ok = e->impl_->Build(e->config_.gpu_model.bytes, /*use_gpu=*/true, e->config_.num_threads, &why);
    if (ok) e->backend_ = Backend::kGpu;
    else e->gpu_fallback_reason_ = why;
  } else {
    e->gpu_fallback_reason_ = e->config_.allow_gpu ? "no FP16 model" : "GPU disabled";
  }
  if (!ok) {
    ok = e->impl_->Build(e->config_.cpu_model.bytes, /*use_gpu=*/false, e->config_.num_threads, &why);
    e->backend_ = Backend::kCpu;
  }
  if (!ok) { *error = why; return nullptr; }

  // Check the I/O contract before trusting the model.
  const TfLiteTensor* in = TfLiteInterpreterGetInputTensor(e->impl_->interpreter, 0);
  const TfLiteTensor* out = TfLiteInterpreterGetOutputTensor(e->impl_->interpreter, 0);
  const int s = e->config_.input_size;
  if (TfLiteTensorType(in) != kTfLiteFloat32 || TfLiteTensorNumDims(in) != 4 || TfLiteTensorDim(in, 1) != s ||
      TfLiteTensorDim(in, 2) != s || TfLiteTensorDim(in, 3) != 3) {
    *error = "unexpected input tensor";
    return nullptr;
  }
  if (TfLiteTensorType(out) != kTfLiteFloat32 ||
      TfLiteTensorByteSize(out) != sizeof(float) * static_cast<size_t>(e->config_.num_labels)) {
    *error = "unexpected output tensor";
    return nullptr;
  }

  const ModelFile& used = e->backend_ == Backend::kGpu ? e->config_.gpu_model : e->config_.cpu_model;
  e->decision_ = {used.temperature, used.reject_below, e->config_.confident_at, e->config_.unknown_index};

  // Warm-up: first invoke compiles GPU kernels / packs XNNPACK weights.
  std::memset(TfLiteTensorData(TfLiteInterpreterGetInputTensor(e->impl_->interpreter, 0)), 0,
              TfLiteTensorByteSize(in));
  const auto t = Clock::now();
  if (TfLiteInterpreterInvoke(e->impl_->interpreter) != kTfLiteOk) { *error = "warm-up invoke failed"; return nullptr; }
  e->warmup_ms_ = MsSince(t);
  return e;
}

QualityResult Engine::CheckQuality(const uint8_t* gray, int w, int h, int row_stride) {
  return gate_.Check(gray, w, h, row_stride);
}

VisionResult Engine::Infer(VisionResult r) {
  const auto t = Clock::now();
  if (TfLiteInterpreterInvoke(impl_->interpreter) != kTfLiteOk) {
    r.decision = Decision::kNotRecognised;  // never guess on a failed run
    return r;
  }
  r.inference_ms = MsSince(t);
  const float* logits = static_cast<const float*>(TfLiteTensorData(TfLiteInterpreterGetOutputTensor(impl_->interpreter, 0)));
  r.decision = Decide(logits, static_cast<size_t>(config_.num_labels), decision_, r.probs.data(), &r.top);
  r.top_prob = r.probs[r.top];
  // Severity only for an answered disease label, on the crop the model saw.
  const bool answered = r.decision == Decision::kConfident || r.decision == Decision::kUncertain;
  if (answered && r.top < static_cast<int>(config_.severity_labels.size()) && config_.severity_labels[r.top]) {
    const auto s = Clock::now();
    const int n = config_.input_size;
    r.severity = severity_.Estimate(pre_.last_crop().data(), n, n, n * 3);
    r.severity_ms = MsSince(s);
  }
  return r;
}

VisionResult Engine::AnalyzeYuv(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, int y_row_stride,
                                int uv_row_stride, int uv_pixel_stride, int rotation_degrees) {
  VisionResult r;
  auto t = Clock::now();
  r.quality = gate_.Check(y, w, h, y_row_stride);  // Y plane is the luma image
  r.quality_ms = MsSince(t);
  if (r.quality.verdict != QualityVerdict::kOk) return r;  // retake: the model never runs
  t = Clock::now();
  auto* input = static_cast<float*>(TfLiteTensorData(TfLiteInterpreterGetInputTensor(impl_->interpreter, 0)));
  pre_.FromYuv420(y, u, v, w, h, y_row_stride, uv_row_stride, uv_pixel_stride, rotation_degrees, input);
  r.preprocess_ms = MsSince(t);
  return Infer(r);
}

VisionResult Engine::AnalyzeRgb(const uint8_t* rgb, int w, int h, int row_stride, int rotation_degrees) {
  VisionResult r;
  auto t = Clock::now();
  r.quality = gate_.CheckRgb(rgb, w, h, row_stride);
  r.quality_ms = MsSince(t);
  if (r.quality.verdict != QualityVerdict::kOk) return r;
  t = Clock::now();
  auto* input = static_cast<float*>(TfLiteTensorData(TfLiteInterpreterGetInputTensor(impl_->interpreter, 0)));
  pre_.FromRgb(rgb, w, h, row_stride, rotation_degrees, input);
  r.preprocess_ms = MsSince(t);
  return Infer(r);
}

}  // namespace farm
