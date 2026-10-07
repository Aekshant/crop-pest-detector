package com.farm.app.vision

import java.nio.ByteBuffer

/** JNI entry points of the C++ core (android/app/src/main/cpp/farm_jni.cpp). */
internal object VisionNative {
    init {
        System.loadLibrary("farm_jni")
    }

    // Error codes returned by every native call (keep in sync with farm_jni.cpp).
    const val OK = 0
    const val INVALID_HANDLE = 1
    const val INVALID_ARGUMENT = 2
    const val BUFFER_NOT_DIRECT = 3
    const val NATIVE_EXCEPTION = 4
    const val OUT_OF_MEMORY = 5
    const val INFERENCE_FAILED = 6

    // Float result layout (keep in sync with farm_jni.cpp).
    const val DECISION = 0
    const val TOP = 1
    const val TOP_PROB = 2
    const val SHARPNESS = 3
    const val BRIGHTNESS = 4
    const val QUALITY_VERDICT = 5
    const val QUALITY_MS = 6
    const val PREPROCESS_MS = 7
    const val INFERENCE_MS = 8
    const val SEVERITY_FRACTION = 9
    const val SEVERITY_LEVEL = 10
    const val SEVERITY_MS = 11
    const val PROBS = 12

    /** params: tempGpu, rejectGpu, tempCpu, rejectCpu, confidentAt, minSharpness, minBrightness, maxBrightness. */
    external fun nativeCreate(
        fp16: ByteBuffer?, int8: ByteBuffer, params: FloatArray, numLabels: Int, unknownIndex: Int,
        inputSize: Int, threads: Int, allowGpu: Boolean, severityLabels: BooleanArray,
    ): Long

    external fun nativeRelease(handle: Long): Int
    external fun nativeInfo(handle: Long, out: FloatArray): Int
    external fun nativeGpuFallbackReason(handle: Long): String

    external fun nativeAnalyzeYuv(
        handle: Long, y: ByteBuffer, u: ByteBuffer, v: ByteBuffer, width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int, rotation: Int, numLabels: Int, out: FloatArray,
    ): Int

    external fun nativeAnalyzeRgb(
        handle: Long, rgb: ByteArray?, width: Int, height: Int, rotation: Int, numLabels: Int, out: FloatArray,
    ): Int

    external fun nativeCheckQuality(handle: Long, gray: ByteBuffer?, width: Int, height: Int, rowStride: Int, out: FloatArray): Int

    /** Test hook: makes native code throw (1 runtime_error, 2 bad_alloc, 3 non-std) to prove it becomes a code. */
    external fun nativeForceError(kind: Int): Int
}
