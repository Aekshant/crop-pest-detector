package com.farm.app.ui.laya

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ChatMessage(val fromUser: Boolean, val text: String)

data class LayaUiState(
    val messages: List<ChatMessage> = emptyList(),
    val draft: String = "",
    val thinking: Boolean = false,
)

/**
 * Chat state for Laya. The reply is a placeholder until the on-device SLM and the
 * execution router exist; [reply] is the seam where they plug in.
 */
class LayaViewModel(
    private val reply: suspend (String) -> String = { NOT_CONNECTED },
) : ViewModel() {
    private val _state = MutableStateFlow(LayaUiState())
    val state: StateFlow<LayaUiState> = _state.asStateFlow()

    fun onDraftChange(text: String) = _state.update { it.copy(draft = text) }

    fun send() {
        val text = _state.value.draft.trim()
        if (text.isEmpty() || _state.value.thinking) return
        _state.update {
            it.copy(messages = it.messages + ChatMessage(fromUser = true, text = text), draft = "", thinking = true)
        }
        viewModelScope.launch {
            val answer = reply(text)
            _state.update {
                it.copy(messages = it.messages + ChatMessage(fromUser = false, text = answer), thinking = false)
            }
        }
    }

    companion object {
        const val NOT_CONNECTED = "Laya is not connected yet. / लाया अभी जुड़ी नहीं है।"
    }
}
