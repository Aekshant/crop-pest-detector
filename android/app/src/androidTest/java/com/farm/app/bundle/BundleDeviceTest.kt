package com.farm.app.bundle

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.farm.app.BuildConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The whole update flow on a phone, against a real bundle (ml/bundle/build_bundle.py)
 * served from the dev machine:
 *   python3 -m http.server 8099 --directory dist/bundles   # and: adb reverse tcp:8099 tcp:8099
 *   adb shell am instrument -w -e class com.farm.app.bundle.BundleDeviceTest \
 *       -e bundleUrl http://127.0.0.1:8099 [-e expect smoke_fail] [-e keep true] com.farm.app.test/androidx.test.runner.AndroidJUnitRunner
 * Uses its own store under files/test-bundles unless -e keep true (then the app's real store, left installed).
 */
@RunWith(AndroidJUnit4::class)
class BundleDeviceTest {
    @Test fun updateFromServedBundle() {
        val args = InstrumentationRegistry.getArguments()
        val url = args.getString("bundleUrl")
        assumeTrue("pass -e bundleUrl", url != null)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        // -e keep true: install into the app's real store and leave it there (to try the app on it).
        val keep = args.getString("keep") == "true"
        val store = if (keep) BundleWorker.store(context) else BundleStore(File(context.filesDir, "test-bundles").apply { deleteRecursively() })
        val tier = DeviceTier.current(context)
        val updater = BundleUpdater(
            HttpBundleSource(url!!), store, BundleManifest.keys(BuildConfig.BUNDLE_PUBLIC_KEYS), BuildConfig.VERSION_CODE,
            BundleWorker.SUPPORTED_TOOL_SCHEMAS, tier, BundleSmoke(context, tier),
        )
        val t0 = System.nanoTime()
        val result = updater.update()
        val seconds = (System.nanoTime() - t0) / 1e9
        println("BUNDLE_DEVICE_TEST tier=$tier result=$result seconds=%.1f".format(seconds))
        if (args.getString("expect") == "smoke_fail") {
            assertTrue(result.toString(), result is UpdateResult.SmokeFailed)
            assertEquals(null, store.current())
        } else {
            assertTrue(result.toString(), result is UpdateResult.Updated || (keep && result is UpdateResult.UpToDate))
            val version = (result as? UpdateResult.Updated)?.version ?: (result as UpdateResult.UpToDate).version
            assertEquals(version, store.current())
            assertEquals(UpdateResult.UpToDate(version), updater.update())
        }
        if (!keep) store.root.deleteRecursively()
    }
}
