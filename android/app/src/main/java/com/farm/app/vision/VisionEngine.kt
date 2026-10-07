package com.farm.app.vision

import android.content.Context
import android.os.Build
import androidx.camera.core.ImageProxy

enum class Decision { RETAKE, NOT_RECOGNISED, UNCERTAIN, CONFIDENT }
enum class QualityVerdict { OK, BLURRY, TOO_DARK, TOO_BRIGHT }
enum class Backend { GPU, CPU }

/** Outcome of one analysis: a result, or a native error code (never a crash). */
sealed interface VisionOutcome {
    data class Ok(val result: VisionResult) : VisionOutcome
    data class Failed(val code: Int) : VisionOutcome {
        val reason: String
            get() = when (code) {
                VisionNative.INVALID_HANDLE -> "engine closed"
                VisionNative.INVALID_ARGUMENT -> "invalid input"
                VisionNative.BUFFER_NOT_DIRECT -> "camera buffer not direct"
                VisionNative.NATIVE_EXCEPTION -> "native error"
                VisionNative.OUT_OF_MEMORY -> "out of memory"
                VisionNative.INFERENCE_FAILED -> "model run failed"
                else -> "error $code"
            }
    }
}

fun VisionOutcome.resultOrNull(): VisionResult? = (this as? VisionOutcome.Ok)?.result

/** Lesion share of the leaf, from colour segmentation. Always shown as an estimate. */
enum class SeverityLevel { UNKNOWN, LOW, MEDIUM, HIGH }

data class VisionResult(
    val decision: Decision,
    val top: Label?,
    val topProb: Float,
    val probs: FloatArray,
    val sharpness: Float,
    val brightness: Float,
    val quality: QualityVerdict,
    val qualityMs: Float,
    val preprocessMs: Float,
    val inferenceMs: Float,
    val severityLevel: SeverityLevel = SeverityLevel.UNKNOWN,
    val severityFraction: Float = 0f,
    val severityMs: Float = 0f,
) {
    /** Second-best label, shown alongside an uncertain answer. */
    fun runnerUp(labels: List<Label>): Label? =
        probs.indices.filter { it != probs.indices.maxByOrNull { i -> probs[i] } }.maxByOrNull { probs[it] }?.let { labels[it] }
}

/**
 * The C++ vision engine (quality gate, preprocessing, LiteRT inference).
 *
 * Thread-confined: create, use and close it on the same thread
 * ([VisionThread]). Not thread-safe by design: the interpreter and GPU
 * delegate must never be shared across threads.
 */
class VisionEngine private constructor(
    private var handle: Long,
    val config: VisionConfig,
    val backend: Backend,
    val warmupMs: Float,
    val gpuFallbackReason: String,
) : AutoCloseable {
    private val out = FloatArray(VisionNative.PROBS + config.labels.size)

    /** Analyzes a CameraX YUV_420_888 frame without copying its planes. */
    fun analyze(image: ImageProxy): VisionOutcome {
        val (y, u, v) = image.planes
        val code = VisionNative.nativeAnalyzeYuv(
            handle, y.buffer, u.buffer, v.buffer, image.width, image.height,
            y.rowStride, u.rowStride, u.pixelStride, image.imageInfo.rotationDegrees, config.labels.size, out,
        )
        return outcome(code)
    }

    /** Analyzes an RGB8 image (golden tests, gallery photos). */
    fun analyzeRgb(rgb: ByteArray, width: Int, height: Int, rotationDegrees: Int = 0): VisionOutcome =
        outcome(VisionNative.nativeAnalyzeRgb(handle, rgb, width, height, rotationDegrees, config.labels.size, out))

    private fun outcome(code: Int) = if (code == VisionNative.OK) VisionOutcome.Ok(result()) else VisionOutcome.Failed(code)

    private fun result(): VisionResult {
        val top = out[VisionNative.TOP].toInt()
        return VisionResult(
            decision = Decision.entries[out[VisionNative.DECISION].toInt()],
            top = config.labels.getOrNull(top),
            topProb = out[VisionNative.TOP_PROB],
            probs = out.copyOfRange(VisionNative.PROBS, VisionNative.PROBS + config.labels.size),
            sharpness = out[VisionNative.SHARPNESS],
            brightness = out[VisionNative.BRIGHTNESS],
            quality = QualityVerdict.entries[out[VisionNative.QUALITY_VERDICT].toInt()],
            qualityMs = out[VisionNative.QUALITY_MS],
            preprocessMs = out[VisionNative.PREPROCESS_MS],
            inferenceMs = out[VisionNative.INFERENCE_MS],
            severityLevel = SeverityLevel.entries[out[VisionNative.SEVERITY_LEVEL].toInt()],
            severityFraction = out[VisionNative.SEVERITY_FRACTION],
            severityMs = out[VisionNative.SEVERITY_MS],
        )
    }

    /** Idempotent; afterwards every call returns INVALID_HANDLE instead of crashing. */
    override fun close() {
        if (handle != 0L) VisionNative.nativeRelease(handle)
        handle = 0L
    }

    companion object {
        /**
         * XNNPACK worker threads spin-wait at every layer boundary, so asking for
         * more threads than cores slows inference down badly. DeviceProfiler can
         * refine this from its benchmark.
         */
        fun defaultThreads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)

        /**
         * True on Android emulators: their GPU is a software renderer on which the
         * GPU delegate can hang (not just fail) while compiling shaders.
         */
        val isEmulator: Boolean
            get() = Build.FINGERPRINT.startsWith("generic") || Build.FINGERPRINT.contains("emulator") ||
                Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish") || Build.PRODUCT.contains("sdk")

        /**
         * Loads models from assets, picks GPU (FP16) or CPU (INT8), and warms up. Slow: call on [VisionThread].
         * The GPU is not tried on emulators. On phones, DeviceProfiler (later task) should validate the GPU
         * path once with a timeout and pass allowGpu = false where it hangs or gives wrong golden output.
         */
        fun create(context: Context, allowGpu: Boolean = !isEmulator, threads: Int = defaultThreads()): VisionEngine {
            val config = VisionConfig.load(context)
            val params = floatArrayOf(
                config.fp16.temperature, config.fp16.rejectBelow, config.int8.temperature, config.int8.rejectBelow,
                config.confidentAt, config.minSharpness, config.minBrightness, config.maxBrightness,
            )
            val handle = VisionNative.nativeCreate(
                if (allowGpu) VisionConfig.modelBuffer(context, "vision_fp16.tflite") else null,
                VisionConfig.modelBuffer(context, "vision_int8.tflite"),
                params, config.labels.size, config.unknownIndex, config.inputSize, threads, allowGpu,
                BooleanArray(config.labels.size) { config.labels[it].isDisease },
            )
            val info = FloatArray(2)
            check(VisionNative.nativeInfo(handle, info) == VisionNative.OK) { "engine info failed" }
            return VisionEngine(
                handle, config,
                backend = if (info[0] == 0f) Backend.GPU else Backend.CPU,
                warmupMs = info[1],
                gpuFallbackReason = VisionNative.nativeGpuFallbackReason(handle),
            )
        }
    }
}
