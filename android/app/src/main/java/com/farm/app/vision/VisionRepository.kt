package com.farm.app.vision

import android.content.Context
import android.util.Log
import com.farm.app.bundle.ActiveBundle
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

data class EngineStatus(
    val ready: Boolean = false,
    val backend: Backend? = null,
    val warmupMs: Float = 0f,
    val gpuFallbackReason: String = "",
    val modelVersion: String = "",
    val error: String? = null,
    /** Where the model came from: "2026.10.15 (current)", "built-in", ... */
    val source: String = "",
    /** Sources that failed before this one loaded (e.g. a broken current bundle). */
    val skipped: List<String> = emptyList(),
)

/**
 * App-wide owner of the vision engine: created once at app start (with a
 * warm-up run), released when the app is backgrounded (onTrimMemory), and
 * recreated on demand. All engine access happens on [VisionThread].
 */
object VisionRepository {
    private const val TAG = "VisionRepository"
    private var engine: VisionEngine? = null  // touched only on VisionThread
    private var gpuBroken = false             // set when the GPU path fails at run time
    private val _status = MutableStateFlow(EngineStatus())
    val status: StateFlow<EngineStatus> = _status.asStateFlow()

    /** Creates the engine if needed. Must run on [VisionThread]. */
    fun obtain(context: Context): VisionEngine? {
        check(VisionThread.isCurrent()) { "vision engine used off the vision thread" }
        engine?.let { return it }
        val app = context.applicationContext
        val skipped = runCatching { ActiveBundle.rejected(app) }.getOrDefault(emptyList())
            .map { "$it: damaged or not trusted" }.toMutableList()
        var lastError: Exception? = null
        // Downloaded bundles first (current, then previous), then the APK's built-in models.
        for ((name, files) in sources(app)) {
            try {
                val e = VisionEngine.create(app, allowGpu = !gpuBroken && !VisionEngine.isEmulator, files = files)
                engine = e
                _status.value = EngineStatus(true, e.backend, e.warmupMs, e.gpuFallbackReason, e.config.modelVersion, null, name, skipped)
                Log.i(TAG, "engine ready from $name: ${e.backend} warm-up ${"%.1f".format(e.warmupMs)} ms ${e.gpuFallbackReason}")
                return e
            } catch (e: Exception) {
                Log.e(TAG, "engine from $name failed", e)
                skipped += "$name: ${e.message ?: e}"
                lastError = e
            }
        }
        _status.value = EngineStatus(error = lastError?.message ?: "no vision model", skipped = skipped)
        return null
    }

    private fun sources(context: Context): List<Pair<String, VisionFiles>> {
        val bundles = runCatching { ActiveBundle.candidates(context) }.getOrDefault(emptyList()).filter { it.hasVision }
        return bundles.map { it.toString() to VisionFiles.dir(it.dir) } + ("built-in" to VisionFiles.assets(context))
    }

    /** Warm the engine up off the main thread (called at app start). */
    suspend fun warmUp(context: Context) = withContext(VisionThread.dispatcher) { obtain(context) }

    /**
     * A failed model run on the GPU means the delegate is broken on this device:
     * switch to CPU for the rest of the session (design doc: GPU delegate broken
     * -> CPU INT8). Must run on [VisionThread].
     */
    fun onInferenceFailed(context: Context) {
        check(VisionThread.isCurrent())
        val e = engine ?: return
        Log.w(TAG, "inference failed on ${e.backend}")
        if (e.backend == Backend.GPU) {
            gpuBroken = true
            e.close()
            engine = null
            obtain(context)  // recreate on CPU
        }
    }

    /** Frees the model memory (app backgrounded); the next frame recreates it. */
    fun release() {
        VisionThread.executor.execute {
            engine?.close()
            engine = null
            _status.value = _status.value.copy(ready = false)
        }
    }
}
