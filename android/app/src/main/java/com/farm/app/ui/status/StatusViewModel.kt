package com.farm.app.ui.status

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.farm.app.BuildConfig
import com.farm.app.NativeCore
import com.farm.app.bundle.ActiveBundle
import com.farm.app.bundle.DeviceTier
import com.farm.app.kb.KbRepository
import com.farm.app.vision.EngineStatus
import com.farm.app.vision.VisionRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class StatusUiState(
    val appVersion: String = "",
    /** Null while the native library is loading. */
    val coreVersion: String? = null,
    val engine: EngineStatus = EngineStatus(),
    /** Filled by DeviceProfiler, HealthMonitor and NetworkMonitor in later tasks. */
    val deviceTier: String? = null,
    val network: String? = null,
    /** The active model bundle with a KB, e.g. "2026.10.15 (current)"; null: none downloaded. */
    val kbBundle: String? = null,
    /** Result of the on-demand KB check (tap the row): top passage and timings. */
    val kbCheck: String? = null,
)

/**
 * The design doc's debug/status screen: tier, health, network, breaker state and
 * recent router decisions. For now: versions and the vision engine's backend.
 */
private const val KB_CHECK_QUESTION = "agati jhulsa ke dhabbe kaise dikhte hain?"

class StatusViewModel(
    private val coreVersion: suspend () -> String = { NativeCore.version() },
) : ViewModel() {
    /** Reads which bundles are activated (cheap: signature + file sizes, no model loading). */
    fun loadBundles(context: Context) {
        viewModelScope.launch {
            val tier = withContext(Dispatchers.IO) { DeviceTier.current(context) }
            val kb = withContext(Dispatchers.IO) { ActiveBundle.candidates(context, tier).firstOrNull { it.hasKb }?.toString() }
            _state.update { it.copy(deviceTier = tier, kbBundle = kb) }
        }
    }

    /**
     * Loads the KB from the active bundle and runs one search: proves the whole path
     * (bundle files, LiteRT embedder, retriever) works in this build, release included.
     */
    fun checkKb(context: Context) {
        _state.update { it.copy(kbCheck = "…") }
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val t0 = System.nanoTime()
                    val kb = KbRepository.obtain(context) ?: return@runCatching "no KB in an active bundle"
                    val t1 = System.nanoTime()
                    val top = kb.retriever.search(KB_CHECK_QUESTION, kb.model).first()
                    val t2 = System.nanoTime()
                    "\"$KB_CHECK_QUESTION\" -> ${top.source}: ${top.text.take(70)}… (load %.0f ms, search %.0f ms)"
                        .format((t1 - t0) / 1e6, (t2 - t1) / 1e6)
                }.getOrElse { "failed: ${it.javaClass.simpleName}: ${it.message}" }
            }
            Log.i("KbCheck", result)
            _state.update { it.copy(kbCheck = result) }
        }
    }

    private val _state = MutableStateFlow(StatusUiState(appVersion = BuildConfig.VERSION_NAME))
    val state: StateFlow<StatusUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            // Loading the .so and calling JNI is off the main thread.
            val version = withContext(Dispatchers.Default) { coreVersion() }
            _state.update { it.copy(coreVersion = version) }
        }
        viewModelScope.launch {
            VisionRepository.status.collect { s -> _state.update { it.copy(engine = s) } }
        }
    }
}
