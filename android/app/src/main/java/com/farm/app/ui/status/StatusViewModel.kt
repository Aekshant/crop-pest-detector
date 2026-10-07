package com.farm.app.ui.status

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.farm.app.BuildConfig
import com.farm.app.NativeCore
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
    val modelVersion: String = "vision-v3b-lite0",
    /** Filled by DeviceProfiler, HealthMonitor and NetworkMonitor in later tasks. */
    val deviceTier: String? = null,
    val network: String? = null,
)

/**
 * The design doc's debug/status screen: tier, health, network, breaker state and
 * recent router decisions. For now it shows versions and proves the C++ core loads.
 */
class StatusViewModel(
    private val coreVersion: suspend () -> String = { NativeCore.version() },
) : ViewModel() {
    private val _state = MutableStateFlow(StatusUiState(appVersion = BuildConfig.VERSION_NAME))
    val state: StateFlow<StatusUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            // Loading the .so and calling JNI is off the main thread.
            val version = withContext(Dispatchers.Default) { coreVersion() }
            _state.update { it.copy(coreVersion = version) }
        }
    }
}
