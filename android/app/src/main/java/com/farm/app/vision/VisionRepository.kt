package com.farm.app.vision

import android.content.Context
import android.util.Log
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
)

/**
 * App-wide owner of the vision engine: created once at app start (with a
 * warm-up run), released when the app is backgrounded (onTrimMemory), and
 * recreated on demand. All engine access happens on [VisionThread].
 */
object VisionRepository {
    private const val TAG = "VisionRepository"
    private var engine: VisionEngine? = null  // touched only on VisionThread
    private val _status = MutableStateFlow(EngineStatus())
    val status: StateFlow<EngineStatus> = _status.asStateFlow()

    /** Creates the engine if needed. Must run on [VisionThread]. */
    fun obtain(context: Context): VisionEngine? {
        check(VisionThread.isCurrent()) { "vision engine used off the vision thread" }
        engine?.let { return it }
        return try {
            VisionEngine.create(context.applicationContext).also {
                engine = it
                _status.value = EngineStatus(true, it.backend, it.warmupMs, it.gpuFallbackReason, it.config.modelVersion)
                Log.i(TAG, "engine ready: ${it.backend} warm-up ${"%.1f".format(it.warmupMs)} ms ${it.gpuFallbackReason}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "engine creation failed", e)
            _status.value = EngineStatus(error = e.message ?: e.toString())
            null
        }
    }

    /** Warm the engine up off the main thread (called at app start). */
    suspend fun warmUp(context: Context) = withContext(VisionThread.dispatcher) { obtain(context) }

    /** Frees the model memory (app backgrounded); the next frame recreates it. */
    fun release() {
        VisionThread.executor.execute {
            engine?.close()
            engine = null
            _status.value = _status.value.copy(ready = false)
        }
    }
}
