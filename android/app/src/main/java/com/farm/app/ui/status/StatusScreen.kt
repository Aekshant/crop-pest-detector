package com.farm.app.ui.status

import androidx.compose.foundation.clickable
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.farm.app.R
import com.farm.app.ui.ScreenColumn

@Composable
fun StatusScreen(viewModel: StatusViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    LaunchedEffect(Unit) { viewModel.loadBundles(context.applicationContext) }
    val notYet = stringResource(R.string.status_not_measured)
    ScreenColumn(stringResource(R.string.nav_status)) {
        Row(stringResource(R.string.status_app), state.appVersion)
        Row(stringResource(R.string.status_core), state.coreVersion ?: "…")
        val e = state.engine
        Row(
            stringResource(R.string.status_model),
            when {
                e.error != null -> "error: ${e.error}"
                !e.ready -> stringResource(R.string.status_model_loading)
                else -> "${e.modelVersion} · ${e.backend} · warm-up ${"%.0f".format(e.warmupMs)} ms" +
                    if (e.gpuFallbackReason.isNotEmpty()) " · GPU: ${e.gpuFallbackReason}" else ""
            },
        )
        Row(
            stringResource(R.string.status_bundle),
            when {
                e.source.isEmpty() -> "…"
                e.source == "built-in" -> stringResource(R.string.status_bundle_builtin)
                else -> e.source
            } + e.skipped.joinToString("") { "\nskipped $it" },
        )
        Row(
            stringResource(R.string.status_kb),
            (state.kbBundle ?: stringResource(R.string.status_kb_none)) + (state.kbCheck?.let { "\n$it" } ?: ""),
            onClick = state.kbBundle?.let { { viewModel.checkKb(context.applicationContext) } },
        )
        Row(stringResource(R.string.status_tier), state.deviceTier ?: notYet)
        Row(stringResource(R.string.status_network), state.network ?: notYet)
    }
}

@Composable
private fun Row(name: String, value: String, onClick: (() -> Unit)? = null) {
    ListItem(
        headlineContent = { Text(name) },
        supportingContent = { Text(value) },
        modifier = if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier,
    )
    HorizontalDivider()
}
