package com.farm.app.bundle

import android.content.Context
import android.util.Log
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.farm.app.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Checks for a new model bundle (design doc update flow, step 1: "when ONLINE and on
 * unmetered networks (or charging)"). WorkManager constraints can't express "or", so
 * there are two periodic requests; a lock keeps them from running the update together.
 */
class BundleWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        if (BuildConfig.BUNDLE_BASE_URL.isBlank()) return@withContext Result.success()
        val tier = DeviceTier.current(applicationContext)
        val updater = BundleUpdater(
            source = HttpBundleSource(BuildConfig.BUNDLE_BASE_URL),
            store = store(applicationContext),
            trustedKeys = BundleManifest.keys(BuildConfig.BUNDLE_PUBLIC_KEYS),
            appVersionCode = BuildConfig.VERSION_CODE,
            supportedToolSchemas = SUPPORTED_TOOL_SCHEMAS,
            tier = tier,
            smokeTest = BundleSmoke(applicationContext, tier),
        )
        val result = lock.withLock { updater.update() }
        Log.i(TAG, "bundle update: $result")
        when (result) {
            is UpdateResult.Failed -> Result.retry()  // network trouble: resume later
            else -> Result.success()                   // bad signature / incompatible / smoke failure are reported, not retried
        }
    }

    companion object {
        private const val TAG = "BundleWorker"
        private val lock = Mutex()
        val SUPPORTED_TOOL_SCHEMAS = setOf(1)

        fun store(context: Context) = BundleStore(File(context.filesDir, "bundles"))

        fun schedule(context: Context) {
            if (BuildConfig.BUNDLE_BASE_URL.isBlank()) return
            val wm = WorkManager.getInstance(context)
            fun request(constraints: Constraints) = PeriodicWorkRequestBuilder<BundleWorker>(12, TimeUnit.HOURS)
                .setConstraints(constraints).build()
            wm.enqueueUniquePeriodicWork("bundle-update-unmetered", ExistingPeriodicWorkPolicy.KEEP,
                request(Constraints.Builder().setRequiredNetworkType(NetworkType.UNMETERED).build()))
            wm.enqueueUniquePeriodicWork("bundle-update-charging", ExistingPeriodicWorkPolicy.KEEP,
                request(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).setRequiresCharging(true).build()))
        }
    }
}
