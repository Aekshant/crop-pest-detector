package com.farm.app.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeFalse
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A6 done-when: golden images give the expected labels on device, through the
 * real engine (quality gate, C++ preprocessing, LiteRT). Runs once with the GPU
 * delegate allowed (falls back to CPU where the GPU can't run the model) and
 * once forced to CPU/XNNPACK + INT8.
 */
@RunWith(AndroidJUnit4::class)
class GoldenTest {
    private val testContext = InstrumentationRegistry.getInstrumentation().context
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext

    private data class Golden(val name: String, val w: Int, val h: Int, val label: Int, val sharpness: Double)

    private fun golden(): Pair<List<Golden>, Double> {
        val j = JSONObject(testContext.assets.open("golden.json").bufferedReader().use { it.readText() })
        val minSharp = j.getJSONObject("quality_thresholds").getDouble("min_sharpness")
        val imgs = j.getJSONArray("images")
        return List(imgs.length()) {
            val o = imgs.getJSONObject(it)
            Golden(o.getString("name"), o.getInt("width"), o.getInt("height"), o.getInt("label_index"), o.getDouble("sharpness"))
        } to minSharp
    }

    private fun check(allowGpu: Boolean) {
        val (images, minSharp) = golden()
        assertTrue(images.size >= 5)
        VisionEngine.create(appContext, allowGpu = allowGpu).use { engine ->
            Log.i(TAG, "backend=${engine.backend} warmup=${engine.warmupMs} ms gpuFallback='${engine.gpuFallbackReason}'")
            for (g in images) {
                val rgb = testContext.assets.open("${g.name}.rgb").use { it.readBytes() }
                val r = requireNotNull(engine.analyzeRgb(rgb, g.w, g.h)) { "${g.name}: analyze failed" }
                Log.i(TAG, "${engine.backend} ${g.name}: ${r.decision} ${r.top?.id} p=${"%.3f".format(r.topProb)} " +
                    "sharp=${"%.4f".format(r.sharpness)} gate=${r.qualityMs}ms prep=${r.preprocessMs}ms model=${r.inferenceMs}ms " +
                    "severity=${r.severityLevel} ${"%.1f".format(r.severityFraction * 100)}% in ${r.severityMs}ms")
                // Severity only for disease labels.
                if (r.top?.isDisease != true) assertEquals(g.name, SeverityLevel.UNKNOWN, r.severityLevel)
                // Same sharpness as the Python reference (the C++ gate is bit-exact on host).
                assertEquals(g.name, g.sharpness, r.sharpness.toDouble(), 1e-4)
                if (g.sharpness < minSharp) {
                    assertEquals(g.name, Decision.RETAKE, r.decision)
                } else {
                    assertEquals(g.name, engine.config.labels[g.label].id, r.top?.id)
                    assertTrue("${g.name}: ${r.decision}", r.decision == Decision.CONFIDENT || r.decision == Decision.UNCERTAIN ||
                        engine.config.labels[g.label].id == "unknown")
                }
            }
        }
    }

    @Test fun goldenLabelsWithGpuAllowed() {
        // Emulator GPUs are software renderers where the GPU delegate can hang.
        assumeFalse("GPU delegate is not exercised on emulators", VisionEngine.isEmulator)
        check(allowGpu = true)
    }

    @Test fun goldenLabelsOnCpuInt8() = check(allowGpu = false)

    /** A4 done-when: quality gate < 10 ms per 640x480 frame on device. */
    @Test fun qualityGateUnder10ms() {
        val engine = VisionEngine.create(appContext, allowGpu = false)
        try {
            val y = ByteBuffer.allocateDirect(640 * 480).order(ByteOrder.nativeOrder())
            for (i in 0 until 640 * 480) y.put(i, ((i * 7919) % 251).toByte())
            val out = FloatArray(4)
            val handleField = VisionEngine::class.java.getDeclaredField("handle").apply { isAccessible = true }
            val handle = handleField.getLong(engine)
            repeat(3) { VisionNative.nativeCheckQuality(handle, y, 640, 480, 640, out) }
            val times = FloatArray(20) {
                assertTrue(VisionNative.nativeCheckQuality(handle, y, 640, 480, 640, out))
                out[3]
            }.sorted()
            Log.i(TAG, "quality gate 640x480 median ${times[10]} ms, max ${times.last()} ms")
            assertTrue("median ${times[10]} ms", times[10] < 10f)
        } finally {
            engine.close()
        }
    }

    /** Inference time by thread count on this device (logged, for DeviceProfiler tuning). */
    @Test fun inferenceTimeByThreads() {
        val g = golden().first.first { it.sharpness >= golden().second }
        val rgb = testContext.assets.open("${g.name}.rgb").use { it.readBytes() }
        for (threads in listOf(1, 2, 4)) {
            VisionEngine.create(appContext, allowGpu = false, threads = threads).use { engine ->
                repeat(2) { engine.analyzeRgb(rgb, g.w, g.h) }
                val ms = List(5) { engine.analyzeRgb(rgb, g.w, g.h)!!.inferenceMs }.sorted()
                Log.i(TAG, "threads=$threads cores=${Runtime.getRuntime().availableProcessors()} inference median ${ms[2]} ms")
            }
        }
    }

    companion object {
        private const val TAG = "GoldenTest"
    }
}
