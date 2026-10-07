package com.farm.app.ui.laya

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.padding
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.farm.app.R
import com.farm.app.ui.ScreenColumn

@Composable
fun LayaScreen(viewModel: LayaViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    ScreenColumn(stringResource(R.string.nav_laya)) {
        LazyColumn(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.messages.isEmpty()) item { Text(stringResource(R.string.laya_hint)) }
            items(state.messages) { msg ->
                Box(Modifier.fillMaxWidth(), contentAlignment = if (msg.fromUser) Alignment.CenterEnd else Alignment.CenterStart) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = if (msg.fromUser) MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.surfaceVariant,
                    ),
                ) { Text(msg.text, modifier = Modifier.padding(12.dp)) }
                }
            }
            if (state.thinking) item { Text("…") }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = state.draft,
                onValueChange = viewModel::onDraftChange,
                placeholder = { Text(stringResource(R.string.laya_placeholder)) },
                modifier = Modifier.weight(1f),
                singleLine = true,
            )
            Button(onClick = viewModel::send, enabled = state.draft.isNotBlank() && !state.thinking) {
                Text(stringResource(R.string.laya_send))
            }
        }
    }
}
