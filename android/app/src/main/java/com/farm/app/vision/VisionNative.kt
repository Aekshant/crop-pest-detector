package com.farm.app.vision

import java.nio.ByteBuffer

/** JNI entry points of the C++ core (android/app/src/main/cpp/farm_jni.cpp). */
internal object VisionNative {
    init {
        System.loadLibrary("farm_jni")
    }

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
    const val PROBS = 9

    /** params: tempGpu, rejectGpu, tempCpu, rejectCpu, confidentAt, minSharpness, minBrightness, maxBrightness. */
    external fun nativeCreate(
        fp16: ByteBuffer?, int8: ByteBuffer, params: FloatArray, numLabels: Int, unknownIndex: Int,
        inputSize: Int, threads: Int, allowGpu: Boolean,
    ): Long

    external fun nativeRelease(handle: Long)
    external fun nativeInfo(handle: Long): FloatArray
    external fun nativeGpuFallbackReason(handle: Long): String

    external fun nativeAnalyzeYuv(
        handle: Long, y: ByteBuffer, u: ByteBuffer, v: ByteBuffer, width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int, rotation: Int, numLabels: Int, out: FloatArray,
    ): Boolean

    external fun nativeAnalyzeRgb(
        handle: Long, rgb: ByteArray, width: Int, height: Int, rotation: Int, numLabels: Int, out: FloatArray,
    ): Boolean

    external fun nativeCheckQuality(handle: Long, gray: ByteBuffer, width: Int, height: Int, rowStride: Int, out: FloatArray): Boolean
}
