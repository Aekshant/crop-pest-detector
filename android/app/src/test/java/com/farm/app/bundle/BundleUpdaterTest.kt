package com.farm.app.bundle

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class BundleUpdaterTest {
    @get:Rule val tmp = TemporaryFolder()
    private lateinit var kit: BundleTestKit
    private lateinit var store: BundleStore
    private var smokeCalls = 0
    private var smokeVerdict: String? = null

    private val v1 = mapOf("vision.tflite" to BundleTestKit.data(300_000, 1), "labels.json" to BundleTestKit.data(500, 2),
                           "laya.gguf" to BundleTestKit.data(50_000, 3))

    @Before fun setUp() {
        kit = BundleTestKit(tmp.newFolder("server"))
        store = BundleStore(tmp.newFolder("phone"))
    }

    @After fun tearDown() = kit.close()

    private fun updater(appVersion: Int = 1, tier: String = "A") = BundleUpdater(
        HttpBundleSource(kit.baseUrl), store, kit.trusted, appVersion, setOf(1), tier,
    ) { dir, _ -> smokeCalls++; assertTrue(File(dir, "labels.json").exists()); smokeVerdict }

    @Test fun installsVerifiesAndThenIsUpToDate() {
        kit.publish("v1", v1, tiers = mapOf("laya.gguf" to listOf("A", "B")))
        val r = updater().update()
        assertEquals(UpdateResult.Updated("v1", 350_500, 0), r)
        assertEquals("v1", store.current())
        v1.forEach { (name, data) -> assertTrue(name, File(store.dir("v1"), name).readBytes().contentEquals(data)) }
        assertTrue(File(store.dir("v1"), BundleUpdater.MANIFEST).exists())
        assertEquals(UpdateResult.UpToDate("v1"), updater().update())
        assertEquals(1, smokeCalls)
    }

    @Test fun downloadsOnlyChangedArtifactsAndKeepsPrevious() {
        kit.publish("v1", v1)
        updater().update()
        val changed = v1 + ("labels.json" to BundleTestKit.data(600, 9))
        kit.publish("v2", changed)
        kit.bytesServed.set(0)
        assertEquals(UpdateResult.Updated("v2", 600, 2), updater().update())
        assertEquals(600L, kit.bytesServed.get())
        assertEquals("v2", store.current())
        assertEquals("v1", store.previous())
        kit.publish("v3", changed + ("labels.json" to BundleTestKit.data(700, 10)))
        updater().update()
        assertEquals(setOf("v2", "v3"), File(store.root, "versions").list()!!.toSet())  // v1 cleaned up
    }

    @Test fun resumesAnInterruptedDownload() {
        kit.publish("v1", v1)
        kit.cutAfter = 100_000  // the first artifact download drops after 100 kB
        val first = updater().update()
        assertTrue(first.toString(), first is UpdateResult.Failed)
        assertEquals(null, store.current())
        val partial = store.staging("v1").listFiles()!!.single { it.name.endsWith(".part") }
        assertEquals(100_000L, partial.length())
        kit.bytesServed.set(0)
        val second = updater().update()
        assertTrue(second.toString(), second is UpdateResult.Updated)
        assertEquals(350_500L - 100_000L, kit.bytesServed.get())  // resumed, not restarted
        v1.forEach { (name, data) -> assertTrue(name, File(store.dir("v1"), name).readBytes().contentEquals(data)) }
    }

    @Test fun restartsCleanlyWhenTheServerIgnoresRange() {
        kit.publish("v1", v1)
        kit.cutAfter = 100_000
        updater().update()
        kit.ignoreRange = true
        assertTrue(updater().update() is UpdateResult.Updated)
        v1.forEach { (name, data) -> assertTrue(name, File(store.dir("v1"), name).readBytes().contentEquals(data)) }
    }

    @Test fun corruptArtifactIsRejectedAndNothingActivated() {
        kit.publish("v1", v1, corrupt = "labels.json")
        val r = updater().update()
        assertTrue(r.toString(), r is UpdateResult.Failed && "sha256" in r.reason)
        assertEquals(null, store.current())
    }

    @Test fun incompatibleBundlesAreSkipped() {
        kit.publish("v1", v1, minAppVersion = 5)
        assertTrue(updater(appVersion = 4).update() is UpdateResult.Incompatible)
        kit.publish("v1", v1, toolSchema = 2)
        assertTrue(updater().update() is UpdateResult.Incompatible)
        assertEquals(0L, kit.bytesServed.get())
        assertEquals(null, store.current())
    }

    @Test fun tierDecidesWhichArtifactsAreFetched() {
        kit.publish("v1", v1, tiers = mapOf("laya.gguf" to listOf("A", "B")))
        assertEquals(UpdateResult.Updated("v1", 300_500, 0), updater(tier = "C").update())
        assertFalse(File(store.dir("v1"), "laya.gguf").exists())
    }

    @Test fun failedSmokeTestDiscardsTheCandidate() {
        kit.publish("v1", v1)
        updater().update()
        kit.publish("v2", v1 + ("labels.json" to BundleTestKit.data(600, 9)))
        smokeVerdict = "golden 02_late_blight: got healthy"
        val r = updater().update()
        assertEquals(UpdateResult.SmokeFailed("v2", "golden 02_late_blight: got healthy"), r)
        assertEquals("v1", store.current())
        assertFalse(store.staging("v2").exists())
    }

    @Test fun badSignatureDownloadsNothing() {
        kit.publish("v1", v1)
        File(kit.root, "manifest.json").writeText(kit.envelope("{}".toByteArray()).replace("ed25519:", "ed25519:AA"))
        assertTrue(updater().update() is UpdateResult.BadSignature)
        assertEquals(0L, kit.bytesServed.get())
    }

    @Test fun deviceTiers() {
        assertEquals("A", DeviceTier.of(7_598_684L * 1024))  // the POCO M6 Plus: "8 GB"
        assertEquals("B", DeviceTier.of(5_800_000_000L))
        assertEquals("C", DeviceTier.of(3_700_000_000L))
    }
}
