// JNI bridge between Kotlin (com.farm.app.vision.VisionNative) and the C++ core.
//
// The engine is created once and handed to Kotlin as an opaque handle (a
// pointer in a jlong). Every call after creation must come from the same
// thread (the app's single "vision" thread). Camera planes arrive as direct
// ByteBuffers (no copy); results go into a float[] Kotlin allocates once.
// C++ exceptions never cross into the JVM: they become error results.

#include <android/log.h>
#include <jni.h>

#include <chrono>
#include <exception>
#include <memory>
#include <string>

#include "farm/core.h"
#include "farm/engine.h"

namespace {

constexpr const char* kTag = "farm_jni";

// Layout of the float[] result (keep in sync with VisionNative.kt).
enum Out : int {
  kDecision = 0, kTop, kTopProb, kSharpness, kBrightness, kQualityVerdict,
  kQualityMs, kPreprocessMs, kInferenceMs, kProbs  // probs follow
};

farm::Engine* FromHandle(jlong h) { return reinterpret_cast<farm::Engine*>(h); }

std::vector<uint8_t> CopyBuffer(JNIEnv* env, jobject buf) {
  auto* p = static_cast<uint8_t*>(env->GetDirectBufferAddress(buf));
  const jlong n = env->GetDirectBufferCapacity(buf);
  if (!p || n <= 0) return {};
  return std::vector<uint8_t>(p, p + n);
}

void ThrowIllegalState(JNIEnv* env, const std::string& msg) {
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls) env->ThrowNew(cls, msg.c_str());
}

bool WriteResult(JNIEnv* env, const farm::VisionResult& r, int num_labels, jfloatArray out) {
  const jsize n = env->GetArrayLength(out);
  if (n < kProbs + num_labels) return false;
  float* o = env->GetFloatArrayElements(out, nullptr);
  o[kDecision] = static_cast<float>(r.decision);
  o[kTop] = static_cast<float>(r.top);
  o[kTopProb] = r.top_prob;
  o[kSharpness] = r.quality.sharpness;
  o[kBrightness] = r.quality.brightness;
  o[kQualityVerdict] = static_cast<float>(r.quality.verdict);
  o[kQualityMs] = r.quality_ms;
  o[kPreprocessMs] = r.preprocess_ms;
  o[kInferenceMs] = r.inference_ms;
  for (int i = 0; i < num_labels; ++i) o[kProbs + i] = r.probs[i];
  env->ReleaseFloatArrayElements(out, o, 0);
  return true;
}

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_farm_app_NativeCore_version(JNIEnv* env, jobject) {
  return env->NewStringUTF(farm::version().c_str());
}

// params: [tempGpu, rejectGpu, tempCpu, rejectCpu, confidentAt, minSharpness, minBrightness, maxBrightness]
JNIEXPORT jlong JNICALL Java_com_farm_app_vision_VisionNative_nativeCreate(
    JNIEnv* env, jobject, jobject fp16, jobject int8, jfloatArray params, jint num_labels, jint unknown_index,
    jint input_size, jint threads, jboolean allow_gpu) {
  try {
    float p[8];
    if (env->GetArrayLength(params) < 8) { ThrowIllegalState(env, "bad params"); return 0; }
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
    std::string error;
    auto engine = farm::Engine::Create(std::move(c), &error);
    if (!engine) { ThrowIllegalState(env, "vision engine: " + error); return 0; }
    __android_log_print(ANDROID_LOG_INFO, kTag, "engine ready: backend=%s warmup=%.1f ms%s%s",
                        engine->backend() == farm::Backend::kGpu ? "gpu" : "cpu", engine->warmup_ms(),
                        engine->gpu_fallback_reason().empty() ? "" : " gpu_fallback=",
                        engine->gpu_fallback_reason().c_str());
    return reinterpret_cast<jlong>(engine.release());
  } catch (const std::exception& e) {
    ThrowIllegalState(env, std::string("vision engine: ") + e.what());
  } catch (...) {
    ThrowIllegalState(env, "vision engine: unknown native error");
  }
  return 0;
}

JNIEXPORT void JNICALL Java_com_farm_app_vision_VisionNative_nativeRelease(JNIEnv*, jobject, jlong h) {
  delete FromHandle(h);
}

// Returns [backend (0 gpu, 1 cpu), warmupMs]
JNIEXPORT jfloatArray JNICALL Java_com_farm_app_vision_VisionNative_nativeInfo(JNIEnv* env, jobject, jlong h) {
  jfloatArray a = env->NewFloatArray(2);
  const float v[2] = {static_cast<float>(FromHandle(h)->backend()), FromHandle(h)->warmup_ms()};
  env->SetFloatArrayRegion(a, 0, 2, v);
  return a;
}

JNIEXPORT jstring JNICALL Java_com_farm_app_vision_VisionNative_nativeGpuFallbackReason(JNIEnv* env, jobject, jlong h) {
  return env->NewStringUTF(FromHandle(h)->gpu_fallback_reason().c_str());
}

JNIEXPORT jboolean JNICALL Java_com_farm_app_vision_VisionNative_nativeAnalyzeYuv(
    JNIEnv* env, jobject, jlong h, jobject y, jobject u, jobject v, jint w, jint ht, jint y_row_stride,
    jint uv_row_stride, jint uv_pixel_stride, jint rotation, jint num_labels, jfloatArray out) {
  try {
    auto* yp = static_cast<uint8_t*>(env->GetDirectBufferAddress(y));
    auto* up = static_cast<uint8_t*>(env->GetDirectBufferAddress(u));
    auto* vp = static_cast<uint8_t*>(env->GetDirectBufferAddress(v));
    if (!yp || !up || !vp) return JNI_FALSE;
    const auto r = FromHandle(h)->AnalyzeYuv(yp, up, vp, w, ht, y_row_stride, uv_row_stride, uv_pixel_stride, rotation);
    return WriteResult(env, r, num_labels, out) ? JNI_TRUE : JNI_FALSE;
  } catch (...) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "AnalyzeYuv threw");
    return JNI_FALSE;
  }
}

JNIEXPORT jboolean JNICALL Java_com_farm_app_vision_VisionNative_nativeAnalyzeRgb(
    JNIEnv* env, jobject, jlong h, jbyteArray rgb, jint w, jint ht, jint rotation, jint num_labels, jfloatArray out) {
  try {
    if (env->GetArrayLength(rgb) < w * ht * 3) return JNI_FALSE;
    jbyte* p = env->GetByteArrayElements(rgb, nullptr);
    const auto r = FromHandle(h)->AnalyzeRgb(reinterpret_cast<uint8_t*>(p), w, ht, w * 3, rotation);
    env->ReleaseByteArrayElements(rgb, p, JNI_ABORT);
    return WriteResult(env, r, num_labels, out) ? JNI_TRUE : JNI_FALSE;
  } catch (...) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "AnalyzeRgb threw");
    return JNI_FALSE;
  }
}

// Quality gate only, on a Y/gray plane: [sharpness, brightness, verdict, ms]
JNIEXPORT jboolean JNICALL Java_com_farm_app_vision_VisionNative_nativeCheckQuality(
    JNIEnv* env, jobject, jlong h, jobject gray, jint w, jint ht, jint row_stride, jfloatArray out) {
  try {
    auto* p = static_cast<uint8_t*>(env->GetDirectBufferAddress(gray));
    if (!p || env->GetArrayLength(out) < 4) return JNI_FALSE;
    const auto t = std::chrono::steady_clock::now();
    const auto q = FromHandle(h)->CheckQuality(p, w, ht, row_stride);
    const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t).count();
    const float v[4] = {q.sharpness, q.brightness, static_cast<float>(q.verdict), ms};
    env->SetFloatArrayRegion(out, 0, 4, v);
    return JNI_TRUE;
  } catch (...) {
    return JNI_FALSE;
  }
}

}  // extern "C"
