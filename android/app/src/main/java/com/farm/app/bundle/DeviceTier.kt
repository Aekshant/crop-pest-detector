package com.farm.app.bundle

import android.app.ActivityManager
import android.content.Context

/**
 * Device tier for bundle selection (docs/budget.md): A = 8 GB+, B = 6 GB, C = 4 GB.
 * By RAM only for now; DeviceProfiler (later) should also demote to C when the GPU
 * delegate fails. Marketing "8 GB" phones report ~7.5 GB total, hence the margins.
 */
object DeviceTier {
    fun of(totalRamBytes: Long): String {
        val gb = totalRamBytes / 1e9
        return when {
            gb >= 7.0 -> "A"
            gb >= 5.0 -> "B"
            else -> "C"
        }
    }

    fun current(context: Context): String {
        val am = context.getSystemService(ActivityManager::class.java)
        return of(ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.totalMem)
    }
}
