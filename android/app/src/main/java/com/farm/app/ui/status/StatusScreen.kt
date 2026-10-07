package com.farm.app.ui.status

import androidx.compose.material3.HorizontalDivider
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
fun StatusScreen(viewModel: StatusViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val notYet = stringResource(R.string.status_not_measured)
    ScreenColumn(stringResource(R.string.nav_status)) {
        Row(stringResource(R.string.status_app), state.appVersion)
        Row(stringResource(R.string.status_core), state.coreVersion ?: "…")
        Row(stringResource(R.string.status_model), state.modelVersion)
        Row(stringResource(R.string.status_tier), state.deviceTier ?: notYet)
        Row(stringResource(R.string.status_network), state.network ?: notYet)
    }
}

@Composable
private fun Row(name: String, value: String) {
    ListItem(headlineContent = { Text(name) }, supportingContent = { Text(value) })
    HorizontalDivider()
}
