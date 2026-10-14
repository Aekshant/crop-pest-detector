package com.farm.app.vision

import android.content.Context
import android.os.Build
import android.os.PerformanceHintManager
import android.os.Process

/**
 * ADPF performance hint for the vision thread (Android 12+). Each analyzed frame
 * reports its duration against a target; when frames run long the system raises
 * CPU clocks. Without it, the GPU path leaves the CPU looking idle between frames,
 * the governor lowers clocks, and the CPU-only quality gate runs ~2x slower (S4).
 * Create and use on the vision thread.
 */
internal class FrameHint private constructor(private val session: Any) : AutoCloseable {
    fun report(nanos: Long) {
        if (Build.VERSION.SDK_INT >= 31) (session as PerformanceHintManager.Session).reportActualWorkDuration(nanos)
    }

    override fun close() {
        if (Build.VERSION.SDK_INT >= 31) (session as PerformanceHintManager.Session).close()
    }

    companion object {
        fun create(context: Context, targetMs: Long): FrameHint? {
            if (Build.VERSION.SDK_INT < 31) return null
            val manager = context.getSystemService(PerformanceHintManager::class.java) ?: return null
            return runCatching { manager.createHintSession(intArrayOf(Process.myTid()), targetMs * 1_000_000L) }
                .getOrNull()?.let { FrameHint(it) }
        }
    }
}
