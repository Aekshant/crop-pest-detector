package com.farm.app.vision

import kotlinx.coroutines.asCoroutineDispatcher
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * The one thread that creates, uses and releases the vision engine. CameraX's
 * analyzer runs on [executor] too, so frames reach the engine on its own thread.
 */
object VisionThread {
    val executor: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "vision") }
    val dispatcher = executor.asCoroutineDispatcher()

    fun isCurrent(): Boolean = Thread.currentThread().name == "vision"
}
