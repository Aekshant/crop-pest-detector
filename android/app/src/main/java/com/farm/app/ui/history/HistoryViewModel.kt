package com.farm.app.ui.history

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class ScanSummary(val id: String, val label: String, val takenAt: String)

data class HistoryUiState(val scans: List<ScanSummary> = emptyList())

/** Scan history; backed by Room in A12. */
class HistoryViewModel : ViewModel() {
    private val _state = MutableStateFlow(HistoryUiState())
    val state: StateFlow<HistoryUiState> = _state.asStateFlow()
}
