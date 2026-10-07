package com.farm.app.ui.scan

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.farm.app.R
import com.farm.app.ui.ScreenColumn
import com.farm.app.vision.Decision
import com.farm.app.vision.Label
import com.farm.app.vision.QualityVerdict
import com.farm.app.vision.VisionRepository
import com.farm.app.vision.VisionResult

@Composable
fun ScanScreen(viewModel: ScanViewModel = viewModel()) {
    val context = LocalContext.current
    val state by viewModel.state.collectAsStateWithLifecycle()
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission(), viewModel::onPermission)
    LaunchedEffect(Unit) {
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        viewModel.onPermission(granted)
        if (!granted) permission.launch(Manifest.permission.CAMERA)
    }

    ScreenColumn(stringResource(R.string.nav_scan)) {
        if (!state.hasPermission) {
            Text(stringResource(R.string.scan_need_permission))
            Button(onClick = { permission.launch(Manifest.permission.CAMERA) }) { Text(stringResource(R.string.scan_allow_camera)) }
            return@ScreenColumn
        }
        Box(Modifier.fillMaxWidth().weight(1f)) {
            CameraPreview(onResult = viewModel::onFrame, modifier = Modifier.fillMaxSize())
            state.live?.let { LiveBadge(it, Modifier.align(Alignment.TopStart).padding(8.dp)) }
        }
        val captured = state.captured
        if (captured != null) {
            ResultCard(captured)
            OutlinedButton(onClick = viewModel::scanAgain, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(R.string.scan_again))
            }
        } else {
            Text(stringResource(R.string.scan_hint))
            Button(onClick = viewModel::takePhoto, enabled = !state.capturing, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(if (state.capturing) R.string.scan_capturing else R.string.scan_take_photo))
            }
        }
    }
}

@Composable
private fun labelName(label: Label?): String {
    if (label == null) return ""
    val hindi = LocalConfiguration.current.locales[0].language == "hi"
    return if (hindi) label.nameHi else label.nameEn
}

@Composable
private fun headline(r: VisionResult): String = when (r.decision) {
    Decision.RETAKE -> stringResource(
        when (r.quality) {
            QualityVerdict.TOO_DARK -> R.string.scan_retake_dark
            QualityVerdict.TOO_BRIGHT -> R.string.scan_retake_bright
            else -> R.string.scan_retake_blurry
        },
    )
    Decision.NOT_RECOGNISED -> stringResource(R.string.scan_not_recognised)
    Decision.UNCERTAIN -> stringResource(R.string.scan_uncertain, labelName(r.top), (r.topProb * 100).toInt())
    Decision.CONFIDENT -> stringResource(R.string.scan_confident, labelName(r.top), (r.topProb * 100).toInt())
}

@Composable
private fun LiveBadge(r: VisionResult, modifier: Modifier) {
    Card(modifier) {
        Text(headline(r), modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp), style = MaterialTheme.typography.labelLarge)
    }
}

@Composable
private fun ResultCard(r: VisionResult) {
    val status by VisionRepository.status.collectAsStateWithLifecycle()
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(headline(r), style = MaterialTheme.typography.titleMedium)
            if (r.decision == Decision.UNCERTAIN) {
                Text(stringResource(R.string.scan_uncertain_hint))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    "${status.backend ?: "-"} · gate ${"%.1f".format(r.qualityMs)} ms · prep ${"%.1f".format(r.preprocessMs)} ms · " +
                        "model ${"%.1f".format(r.inferenceMs)} ms · sharpness ${"%.2f".format(r.sharpness)}",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}
