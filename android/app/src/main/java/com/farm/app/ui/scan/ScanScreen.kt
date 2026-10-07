package com.farm.app.ui.scan

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.farm.app.R
import com.farm.app.ui.ScreenColumn

@Composable
fun ScanScreen(viewModel: ScanViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    ScreenColumn(stringResource(R.string.nav_scan)) {
        Text(stringResource(R.string.scan_hint))
        Button(onClick = {}, enabled = state.cameraReady, modifier = Modifier.fillMaxWidth()) {
            Text(stringResource(R.string.scan_take_photo))
        }
        if (!state.cameraReady) Text(stringResource(R.string.scan_camera_pending))
    }
}
