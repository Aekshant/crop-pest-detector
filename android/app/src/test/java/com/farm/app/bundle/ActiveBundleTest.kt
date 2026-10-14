package com.farm.app.bundle

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ActiveBundleTest {
    @get:Rule val tmp = TemporaryFolder()
    private lateinit var kit: BundleTestKit
    private lateinit var store: BundleStore
    private val files = mapOf("vision.tflite" to BundleTestKit.data(1000, 1), "labels.json" to BundleTestKit.data(100, 2))

    @Before fun setUp() {
        kit = BundleTestKit(tmp.newFolder("server"))
        store = BundleStore(tmp.newFolder("phone"))
    }

    @After fun tearDown() = kit.close()

    private fun install(version: String, content: Map<String, ByteArray> = files) {
        kit.publish(version, content)
        val r = BundleUpdater(HttpBundleSource(kit.baseUrl), store, kit.trusted, 1, setOf(1), "A") { _, _ -> null }.update()
        assertTrue(r.toString(), r is UpdateResult.Updated)
    }

    private fun candidates() = ActiveBundle.candidates(store, kit.trusted, "A").map { it.toString() }

    @Test fun currentThenPrevious() {
        install("v1")
        install("v2", files + ("labels.json" to BundleTestKit.data(100, 3)))
        assertEquals(listOf("v2 (current)", "v1 (previous)"), candidates())
        assertEquals(File(store.dir("v2"), "vision.tflite"), ActiveBundle.candidates(store, kit.trusted, "A")[0].file("vision"))
    }

    @Test fun damagedCurrentFallsBackToPrevious() {
        install("v1")
        install("v2", files + ("labels.json" to BundleTestKit.data(100, 3)))
        File(store.dir("v2"), "vision.tflite").writeBytes(ByteArray(10))  // truncated on disk
        assertEquals(listOf("v1 (previous)"), candidates())
        assertEquals(listOf("v2 (current)"), ActiveBundle.rejected(store, kit.trusted, "A"))
    }

    @Test fun tamperedOrUntrustedManifestIsNotUsed() {
        install("v1")
        val env = File(store.dir("v1"), BundleUpdater.MANIFEST)
        env.writeText(env.readText().replace("ed25519:", "ed25519:AA"))
        assertEquals(emptyList<String>(), candidates())
        install("v2")
        assertEquals(emptyList<String>(), ActiveBundle.candidates(store, emptyMap(), "A"))  // key no longer trusted
    }

    @Test fun nothingInstalled() = assertEquals(emptyList<String>(), candidates())
}
