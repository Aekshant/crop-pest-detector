package com.farm.app.vision

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/**
 * S4 device benchmark: what this phone is, whether golden images give the right
 * labels on each backend, and how long each step takes. Writes a JSON report to
 * the app's external files dir and logs it as one "BENCH_JSON:" line (Firebase
 * Test Lab keeps logcat). Budgets (docs/budget.md): vision inference < 80 ms on
 * GPU (tier A/B), < 150 ms on CPU INT8 (tier C); quality gate < 10 ms (A4).
 */
@RunWith(AndroidJUnit4::class)
class DeviceBenchmark {
    private val testContext = InstrumentationRegistry.getInstrumentation().context
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext

    private data class Golden(val name: String, val w: Int, val h: Int, val label: Int, val rgb: ByteArray)

    private fun goldens(): List<Golden> {
        val imgs = JSONObject(testContext.assets.open("golden.json").bufferedReader().use { it.readText() }).getJSONArray("images")
        return List(imgs.length()) {
            val o = imgs.getJSONObject(it)
            Golden(o.getString("name"), o.getInt("width"), o.getInt("height"), o.getInt("label_index"),
                testContext.assets.open("${o.getString("name")}.rgb").use { s -> s.readBytes() })
        }
    }

    private fun pct(xs: List<Float>, p: Double) = xs.sorted()[((xs.size - 1) * p).toInt()]

    /** Runs one backend configuration on its own thread (engines are thread-confined). */
    private fun measure(name: String, allowGpu: Boolean, threads: Int, goldens: List<Golden>, timeoutS: Long): JSONObject {
        val result = JSONObject().put("config", name).put("threads", threads)
        val exec = Executors.newSingleThreadExecutor()
        val job = exec.submit<JSONObject> {
            val t0 = System.nanoTime()
            VisionEngine.create(appContext, allowGpu = allowGpu, threads = threads).use { e ->
                result.put("backend", e.backend.name).put("create_ms", (System.nanoTime() - t0) / 1e6)
                    .put("warmup_ms", e.warmupMs).put("gpu_fallback", e.gpuFallbackReason)
                var correct = 0
                var gradable = 0
                val inf = mutableListOf<Float>()
                val prep = mutableListOf<Float>()
                val gate = mutableListOf<Float>()
                repeat(3) { e.analyzeRgb(goldens[1].rgb, goldens[1].w, goldens[1].h) }  // settle clocks/caches
                for (round in 0 until 5) {
                    for (g in goldens) {
                        val r = e.analyzeRgb(g.rgb, g.w, g.h) ?: continue
                        gate += r.qualityMs
                        if (r.decision == Decision.RETAKE) continue
                        inf += r.inferenceMs
                        prep += r.preprocessMs
                        if (round == 0) {
                            gradable++
                            if (r.top?.id == e.config.labels[g.label].id) correct++
                        }
                    }
                }
                result.put("golden_correct", correct).put("golden_gradable", gradable)
                    .put("inference_ms_p50", pct(inf, 0.5)).put("inference_ms_p90", pct(inf, 0.9))
                    .put("preprocess_ms_p50", pct(prep, 0.5)).put("gate_ms_p50", pct(gate, 0.5)).put("gate_ms_p90", pct(gate, 0.9))
            }
            result
        }
        return try {
            job.get(timeoutS, TimeUnit.SECONDS)
        } catch (e: TimeoutException) {
            // A GPU delegate can hang instead of failing; report it, don't wait forever.
            result.put("error", "timed out after $timeoutS s (hang)")
        } catch (e: Exception) {
            result.put("error", e.cause?.message ?: e.message ?: e.toString())
        } finally {
            exec.shutdownNow()
        }
    }

    private fun device(): JSONObject {
        val am = appContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mem = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        return JSONObject()
            .put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("device", Build.DEVICE)
            .put("soc", if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE)
            .put("android", Build.VERSION.RELEASE).put("sdk", Build.VERSION.SDK_INT)
            .put("abi", Build.SUPPORTED_ABIS.firstOrNull())
            .put("ram_gb", Math.round(mem.totalMem / 1e8) / 10.0)
            .put("cores", Runtime.getRuntime().availableProcessors())
            .put("emulator", VisionEngine.isEmulator)
            .put("low_ram_device", am.isLowRamDevice)
    }

    @Test fun benchmark() {
        val goldens = goldens()
        val runs = JSONArray()
        if (!VisionEngine.isEmulator) runs.put(measure("gpu_fp16", allowGpu = true, threads = VisionEngine.defaultThreads(), goldens = goldens, timeoutS = 60))
        for (t in listOf(1, 2, 4)) runs.put(measure("cpu_int8_t$t", allowGpu = false, threads = t, goldens = goldens, timeoutS = 180))

        val budget = JSONObject()
        for (i in 0 until runs.length()) {
            val r = runs.getJSONObject(i)
            if (!r.has("inference_ms_p50")) continue
            val limit = if (r.optString("backend") == "GPU") 80.0 else 150.0
            budget.put(r.getString("config"), JSONObject()
                .put("inference_p50_within_budget", r.getDouble("inference_ms_p50") < limit).put("limit_ms", limit)
                .put("gate_p50_within_10ms", r.getDouble("gate_ms_p50") < 10.0))
        }
        val report = JSONObject().put("device", device()).put("model_version", VisionConfig.load(appContext).modelVersion)
            .put("runs", runs).put("budget", budget)
        // Internal copy: `adb exec-out run-as com.farm.app cat files/bench.json` (debug build, no root).
        File(appContext.filesDir, "bench.json").writeText(report.toString(2))
        // External copy: Firebase Test Lab pulls it with --directories-to-pull.
        appContext.getExternalFilesDir(null)?.let { File(it, "bench.json").writeText(report.toString(2)) }
        Log.i(TAG, "BENCH_JSON: $report")
    }

    companion object {
        private const val TAG = "DeviceBenchmark"
    }
}
