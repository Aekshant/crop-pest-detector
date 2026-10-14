package com.farm.app.kb

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class KbBundleTest {
    @get:Rule val tmp = TemporaryFolder()
    private val repo = File("../..").canonicalFile

    private fun manifest(files: Map<String, File>) = JSONObject().put("kb", JSONObject()
        .put("kb_version", "test").put("embedding_model", "e5-small-int8")
        .put("files", JSONObject().apply {
            files.forEach { (k, f) -> put(k, JSONObject().put("file", f.name).put("bytes", f.length()).put("sha256", KbBundle.sha256(f))) }
        })).toString()

    @Test fun intactFilesVerifyAndTamperedOnesDont() {
        val dir = tmp.newFolder()
        val a = File(dir, "kb.sqlite").apply { writeText("pack") }
        val b = File(dir, "model.onnx").apply { writeText("model") }
        val bundle = KbBundle(dir, manifest(mapOf("pack" to a, "embedding_model" to b)))
        assertEquals(emptyList<String>(), bundle.verify())
        b.writeText("modeX")  // same size, different content
        assertEquals(listOf("embedding_model: model.onnx sha256 mismatch"), bundle.verify())
        a.delete()
        assertTrue(bundle.verify().toString(), bundle.verify().any { it == "pack: kb.sqlite missing" })
    }

    @Test fun manifestMatchesTheBuiltFiles() {
        val dist = File(repo, "kb/dist")
        assumeTrue("build kb/dist and run kb/update_manifest.py first", File(dist, "kb_tomato_v1.sqlite").exists())
        val bundle = KbBundle(dist, File(repo, "ml/bundle/manifest_draft.json").readText())
        assertEquals(emptyList<String>(), bundle.verify())
    }
}
