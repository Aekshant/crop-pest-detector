package com.farm.app

import android.content.ComponentCallbacks2
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import com.farm.app.ui.FarmLensApp
import com.farm.app.ui.theme.FarmLensTheme
import com.farm.app.vision.VisionRepository
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Load + warm up the vision model at start, so the first photo is fast.
        lifecycleScope.launch { VisionRepository.warmUp(applicationContext) }
        setContent {
            FarmLensTheme {
                FarmLensApp()
            }
        }
    }

    @Suppress("DEPRECATION")
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // App left the foreground (or memory is low): free the model; the next
        // frame recreates it.
        if (level >= ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN) VisionRepository.release()
    }
}
