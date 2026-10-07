package com.farm.app.ui.scan

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class ScanUiState(
    /** False until CameraX is wired in (A2). */
    val cameraReady: Boolean = false,
)

class ScanViewModel : ViewModel() {
    private val _state = MutableStateFlow(ScanUiState())
    val state: StateFlow<ScanUiState> = _state.asStateFlow()

    fun onCameraReady(ready: Boolean) = _state.update { it.copy(cameraReady = ready) }
}
