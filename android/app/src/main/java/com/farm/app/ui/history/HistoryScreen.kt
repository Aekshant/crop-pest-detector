package com.farm.app.ui.history

import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.ListItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.farm.app.R
import com.farm.app.ui.ScreenColumn

@Composable
fun HistoryScreen(viewModel: HistoryViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    ScreenColumn(stringResource(R.string.nav_history)) {
        if (state.scans.isEmpty()) {
            Text(stringResource(R.string.history_empty))
        } else {
            LazyColumn {
                items(state.scans, key = { it.id }) { scan ->
                    ListItem(headlineContent = { Text(scan.label) }, supportingContent = { Text(scan.takenAt) })
                }
            }
        }
    }
}
