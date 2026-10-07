package com.farm.app.ui.scan

import androidx.lifecycle.ViewModel
import com.farm.app.vision.Decision
import com.farm.app.vision.VisionResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class ScanUiState(
    val hasPermission: Boolean = false,
    /** Latest analyzed camera frame (live mode, 3-5 per second). */
    val live: VisionResult? = null,
    /** Set while a photo burst is collecting frames. */
    val capturing: Boolean = false,
    /** Final answer for the captured photo (best frame of the burst). */
    val captured: VisionResult? = null,
    val framesAnalyzed: Int = 0,
)

/**
 * Scan flow: live analysis, then "Take photo" collects a burst of
 * [BURST_SIZE] frames and keeps the best one (design doc, vision step 2: the
 * sharpest frame that passes the quality gate wins; if none pass, retake).
 */
class ScanViewModel : ViewModel() {
    private val _state = MutableStateFlow(ScanUiState())
    val state: StateFlow<ScanUiState> = _state.asStateFlow()
    private val burst = mutableListOf<VisionResult>()

    fun onPermission(granted: Boolean) = _state.update { it.copy(hasPermission = granted) }

    /** Called from the vision thread for every analyzed frame. */
    fun onFrame(result: VisionResult) {
        var done: VisionResult? = null
        synchronized(burst) {
            if (_state.value.capturing) {
                burst += result
                if (burst.size >= BURST_SIZE) {
                    done = best(burst)
                    burst.clear()
                }
            }
        }
        _state.update { s ->
            val captured = done
            if (captured != null) s.copy(live = result, capturing = false, captured = captured, framesAnalyzed = s.framesAnalyzed + 1)
            else s.copy(live = result, framesAnalyzed = s.framesAnalyzed + 1)
        }
    }

    fun takePhoto() {
        synchronized(burst) { burst.clear() }
        _state.update { it.copy(capturing = true, captured = null) }
    }

    fun scanAgain() = _state.update { it.copy(captured = null, capturing = false) }

    companion object {
        const val BURST_SIZE = 3

        /** Sharpest frame that passed the gate; if none passed, the sharpest overall (a retake). */
        fun best(frames: List<VisionResult>): VisionResult =
            frames.filter { it.decision != Decision.RETAKE }.maxByOrNull { it.sharpness }
                ?: frames.maxBy { it.sharpness }
    }
}
