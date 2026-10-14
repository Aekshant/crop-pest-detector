// JNI bridge between Kotlin (com.farm.app.vision.VisionNative) and the C++ core.
//
// The engine is created once and handed to Kotlin as an opaque handle (a
// pointer in a jlong). Every call after creation must come from the same
// thread (the app's single "vision" thread). Camera planes arrive as direct
// ByteBuffers (no copy); results go into a float[] Kotlin allocates once.
//
// Native failures never crash the app (A8): every entry point runs inside
// Guarded(), which checks the handle and catches all C++ exceptions, and
// returns an ErrorCode instead. Kotlin turns codes into a fallback, not a crash.

#include <android/log.h>
#include <jni.h>
#include <sched.h>

#include <cerrno>

#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "farm/core.h"
#include "farm/engine.h"

namespace {

constexpr const char* kTag = "farm_jni";

// Keep in sync with VisionNative.kt (ErrorCode).
enum ErrorCode : jint {
  kOk = 0,
  kInvalidHandle = 1,
  kInvalidArgument = 2,
  kBufferNotDirect = 3,
  kNativeException = 4,
  kOutOfMemory = 5,
  kInferenceFailed = 6,
};

// Layout of the float[] result (keep in sync with VisionNative.kt).
enum Out : int {
  kDecision = 0, kTop, kTopProb, kSharpness, kBrightness, kQualityVerdict,
  kQualityMs, kPreprocessMs, kInferenceMs,
  kSeverityFraction, kSeverityLevel, kSeverityMs,
  kProbs  // probs follow
};

// Live engines, so a stale or made-up handle is rejected instead of dereferenced.
std::mutex g_mu;
std::unordered_set<farm::Engine*> g_engines;

farm::Engine* Lookup(jlong h) {
  auto* e = reinterpret_cast<farm::Engine*>(h);
  std::lock_guard<std::mutex> lock(g_mu);
  return g_engines.count(e) ? e : nullptr;
}

struct ArgError : std::runtime_error {
  ErrorCode code;
  ArgError(ErrorCode c, const char* what) : std::runtime_error(what), code(c) {}
};

// Runs fn(); maps any failure to an error code and logs it.
template <typename Fn>
jint Guarded(const char* where, Fn&& fn) {
  try {
    return fn();
  } catch (const ArgError& e) {
    __android_log_print(ANDROID_LOG_WARN, kTag, "%s: %s", where, e.what());
    return e.code;
  } catch (const std::bad_alloc&) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s: out of memory", where);
    return kOutOfMemory;
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s: %s", where, e.what());
    return kNativeException;
  } catch (...) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s: unknown native error", where);
    return kNativeException;
  }
}

farm::Engine* RequireEngine(jlong h) {
  farm::Engine* e = Lookup(h);
  if (!e) throw ArgError(kInvalidHandle, "invalid or released engine handle");
  return e;
}

uint8_t* RequireDirect(JNIEnv* env, jobject buf, jlong min_bytes) {
  if (!buf) throw ArgError(kInvalidArgument, "null buffer");
  auto* p = static_cast<uint8_t*>(env->GetDirectBufferAddress(buf));
  if (!p) throw ArgError(kBufferNotDirect, "buffer is not direct");
  if (env->GetDirectBufferCapacity(buf) < min_bytes) throw ArgError(kInvalidArgument, "buffer too small");
  return p;
}

std::vector<uint8_t> CopyBuffer(JNIEnv* env, jobject buf) {
  uint8_t* p = RequireDirect(env, buf, 1);
  return std::vector<uint8_t>(p, p + env->GetDirectBufferCapacity(buf));
}

void ThrowIllegalState(JNIEnv* env, const std::string& msg) {
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls) env->ThrowNew(cls, msg.c_str());
}

jint WriteResult(JNIEnv* env, const farm::VisionResult& r, int num_labels, jfloatArray out) {
  if (!out || env->GetArrayLength(out) < kProbs + num_labels) throw ArgError(kInvalidArgument, "result array too small");
  float o[kProbs + farm::kMaxLabels] = {};
  o[kDecision] = static_cast<float>(r.decision);
  o[kTop] = static_cast<float>(r.top);
  o[kTopProb] = r.top_prob;
  o[kSharpness] = r.quality.sharpness;
  o[kBrightness] = r.quality.brightness;
  o[kQualityVerdict] = static_cast<float>(r.quality.verdict);
  o[kQualityMs] = r.quality_ms;
  o[kPreprocessMs] = r.preprocess_ms;
  o[kInferenceMs] = r.inference_ms;
  o[kSeverityFraction] = r.severity.lesion_fraction;
  o[kSeverityLevel] = static_cast<float>(r.severity.level);
  o[kSeverityMs] = r.severity_ms;
  for (int i = 0; i < num_labels; ++i) o[kProbs + i] = r.probs[i];
  env->SetFloatArrayRegion(out, 0, kProbs + num_labels, o);
  return r.status == farm::Status::kOk ? kOk : kInferenceFailed;
}

void CheckSize(jint w, jint h) {
  if (w <= 0 || h <= 0 || w > 8192 || h > 8192) throw ArgError(kInvalidArgument, "bad image size");
}

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_farm_app_NativeCore_version(JNIEnv* env, jobject) {
  return env->NewStringUTF(farm::version().c_str());
}

// params: [tempGpu, rejectGpu, tempCpu, rejectCpu, confidentAt, minSharpness, minBrightness, maxBrightness]
// Returns a handle, or 0 with an IllegalStateException pending (creation is not a per-frame call).
// Pins the calling thread (and threads it creates later, e.g. XNNPACK workers) to the given
// CPUs. On big.LITTLE phones the vision thread otherwise drifts to little cores while it
// waits for the GPU, and the CPU-only quality gate then runs ~2x slower. 0 or -errno.
JNIEXPORT jint JNICALL Java_com_farm_app_vision_VisionNative_nativePinThread(JNIEnv* env, jobject, jintArray cpus) {
  if (cpus == nullptr) return -EINVAL;
  const jsize n = env->GetArrayLength(cpus);
  jint* ids = env->GetIntArrayElements(cpus, nullptr);
  cpu_set_t set;
  CPU_ZERO(&set);
  for (jsize i = 0; i < n; ++i) {
    if (ids[i] >= 0 && ids[i] < CPU_SETSIZE) CPU_SET(ids[i], &set);
  }
  env->ReleaseIntArrayElements(cpus, ids, JNI_ABORT);
  return sched_setaffinity(0, sizeof(set), &set) == 0 ? 0 : -errno;
}

JNIEXPORT jlong JNICALL Java_com_farm_app_vision_VisionNative_nativeCreate(
    JNIEnv* env, jobject, jobject fp16, jobject int8, jfloatArray params, jint num_labels, jint unknown_index,
    jint input_size, jint threads, jboolean allow_gpu, jbooleanArray severity_labels) {
  jlong handle = 0;
  std::string error;
  const jint code = Guarded("create", [&]() -> jint {
    if (!params || env->GetArrayLength(params) < 8) throw ArgError(kInvalidArgument, "bad params");
    if (num_labels <= 0 || num_labels > farm::kMaxLabels) throw ArgError(kInvalidArgument, "bad label count");
    float p[8];
    env->GetFloatArrayRegion(params, 0, 8, p);
    farm::EngineConfig c;
    if (fp16) c.gpu_model = {CopyBuffer(env, fp16), p[0], p[1]};
    c.cpu_model = {CopyBuffer(env, int8), p[2], p[3]};
    c.confident_at = p[4];
    c.quality = {p[5], p[6], p[7]};
    c.num_labels = num_labels;
    c.unknown_index = unknown_index;
    c.input_size = input_size;
    c.num_threads = threads;
    c.allow_gpu = allow_gpu;
    if (severity_labels) {
      const jsize n = env->GetArrayLength(severity_labels);
      std::vector<jboolean> tmp(static_cast<size_t>(n));
      env->GetBooleanArrayRegion(severity_labels, 0, n, tmp.data());
      c.severity_labels.assign(tmp.begin(), tmp.end());
    }
    auto engine = farm::Engine::Create(std::move(c), &error);
    if (!engine) return kNativeException;
    __android_log_print(ANDROID_LOG_INFO, kTag, "engine ready: backend=%s warmup=%.1f ms%s%s",
                        engine->backend() == farm::Backend::kGpu ? "gpu" : "cpu", engine->warmup_ms(),
                        engine->gpu_fallback_reason().empty() ? "" : " gpu_fallback=",
                        engine->gpu_fallback_reason().c_str());
    farm::Engine* raw = engine.release();
    {
      std::lock_guard<std::mutex> lock(g_mu);
      g_engines.insert(raw);
    }
    handle = reinterpret_cast<jlong>(raw);
    return kOk;
  });
  if (code != kOk && !env->ExceptionCheck())
    ThrowIllegalState(env, "vision engine (code " + std::to_string(code) + ")" + (error.empty() ? "" : ": " + error));
  return handle;
}

JNIEXPORT jint JNICALL Java_com_farm_app_vision_VisionNative_nativeRelease(JNIEnv*, jobject, jlong h) {
  return Guarded("release", [&]() -> jint {
    farm::Engine* e = RequireEngine(h);
    {
      std::lock_guard<std::mutex> lock(g_mu);
      g_engines.erase(e);
    }
    delete e;
    return kOk;
  });
}

// out: [backend (0 gpu, 1 cpu), warmupMs]
JNIEXPORT jint JNICALL Java_com_farm_app_vision_VisionNative_nativeInfo(JNIEnv* env, jobject, jlong h, jfloatArray out) {
  return Guarded("info", [&]() -> jint {
    farm::Engine* e = RequireEngine(h);
    if (!out || env->GetArrayLength(out) < 2) throw ArgError(kInvalidArgument, "info array too small");
    const float v[2] = {static_cast<float>(e->backend()), e->warmup_ms()};
    env->SetFloatArrayRegion(out, 0, 2, v);
    return kOk;
  });
}

JNIEXPORT jstring JNICALL Java_com_farm_app_vision_VisionNative_nativeGpuFallbackReason(JNIEnv* env, jobject, jlong h) {
  std::string reason;
  Guarded("gpu_fallback_reason", [&]() -> jint {
    reason = RequireEngine(h)->gpu_fallback_reason();
    return kOk;
  });
  return env->NewStringUTF(reason.c_str());
}

JNIEXPORT jint JNICALL Java_com_farm_app_vision_VisionNative_nativeAnalyzeYuv(
    JNIEnv* env, jobject, jlong h, jobject y, jobject u, jobject v, jint w, jint ht, jint y_row_stride,
    jint uv_row_stride, jint uv_pixel_stride, jint rotation, jint num_labels, jfloatArray out) {
  return Guarded("analyze_yuv", [&]() -> jint {
    farm::Engine* e = RequireEngine(h);
    CheckSize(w, ht);
    if (y_row_stride < w || uv_pixel_stride < 1 || uv_row_stride < 1) throw ArgError(kInvalidArgument, "bad strides");
    const jlong uv_rows = (ht + 1) / 2, uv_cols = (w + 1) / 2;
    uint8_t* yp = RequireDirect(env, y, static_cast<jlong>(y_row_stride) * (ht - 1) + w);
    uint8_t* up = RequireDirect(env, u, uv_row_stride * (uv_rows - 1) + (uv_cols - 1) * uv_pixel_stride + 1);
    uint8_t* vp = RequireDirect(env, v, uv_row_stride * (uv_rows - 1) + (uv_cols - 1) * uv_pixel_stride + 1);
    const auto r = e->AnalyzeYuv(yp, up, vp, w, ht, y_row_stride, uv_row_stride, uv_pixel_stride, rotation);
    return WriteResult(env, r, num_labels, out);
  });
}

JNIEXPORT jint JNICALL Java_com_farm_app_vision_VisionNative_nativeAnalyzeRgb(
    JNIEnv* env, jobject, jlong h, jbyteArray rgb, jint w, jint ht, jint rotation, jint num_labels, jfloatArray out) {
  return Guarded("analyze_rgb", [&]() -> jint {
    farm::Engine* e = RequireEngine(h);
    CheckSize(w, ht);
    if (!rgb || env->GetArrayLength(rgb) < static_cast<jsize>(w) * ht * 3) throw ArgError(kInvalidArgument, "rgb array too small");
    jbyte* p = env->GetByteArrayElements(rgb, nullptr);
    if (!p) throw std::bad_alloc();
    // Release the array even if analysis throws.
    std::unique_ptr<jbyte, std::function<void(jbyte*)>> guard(p, [&](jbyte* q) { env->ReleaseByteArrayElements(rgb, q, JNI_ABORT); });
    const auto r = e->AnalyzeRgb(reinterpret_cast<uint8_t*>(p), w, ht, w * 3, rotation);
    return WriteResult(env, r, num_labels, out);
  });
}

// Quality gate only, on a Y/gray plane. out: [sharpness, brightness, verdict, ms]
JNIEXPORT jint JNICALL Java_com_farm_app_vision_VisionNative_nativeCheckQuality(
    JNIEnv* env, jobject, jlong h, jobject gray, jint w, jint ht, jint row_stride, jfloatArray out) {
  return Guarded("check_quality", [&]() -> jint {
    farm::Engine* e = RequireEngine(h);
    CheckSize(w, ht);
    if (row_stride < w) throw ArgError(kInvalidArgument, "bad stride");
    uint8_t* p = RequireDirect(env, gray, static_cast<jlong>(row_stride) * (ht - 1) + w);
    if (!out || env->GetArrayLength(out) < 4) throw ArgError(kInvalidArgument, "result array too small");
    const auto t = std::chrono::steady_clock::now();
    const auto q = e->CheckQuality(p, w, ht, row_stride);
    const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t).count();
    const float v[4] = {q.sharpness, q.brightness, static_cast<float>(q.verdict), ms};
    env->SetFloatArrayRegion(out, 0, 4, v);
    return kOk;
  });
}

// Test hook (A8): makes the native side fail on purpose, to prove failures
// come back as error codes. kind: 1 std::runtime_error, 2 std::bad_alloc,
// 3 non-std throw.
JNIEXPORT jint JNICALL Java_com_farm_app_vision_VisionNative_nativeForceError(JNIEnv*, jobject, jint kind) {
  return Guarded("force_error", [&]() -> jint {
    if (kind == 1) throw std::runtime_error("forced error");
    if (kind == 2) throw std::bad_alloc();
    if (kind == 3) throw 42;
    return kOk;
  });
}

}  // extern "C"
