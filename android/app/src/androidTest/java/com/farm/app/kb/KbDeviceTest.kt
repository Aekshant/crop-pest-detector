package com.farm.app.kb

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.farm.app.laya.Grounding
import com.farm.app.laya.Lang
import com.farm.app.laya.LayaContext
import com.farm.app.laya.LayaSession
import com.farm.app.laya.SafetyValidator
import com.farm.app.laya.Templates
import com.farm.app.laya.ToolSchema
import com.farm.app.laya.TurnOutcome
import com.farm.app.laya.Understanding
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The KB on the phone, loaded the way the app loads it (KbRepository, from the installed
 * bundle; install one first: BundleDeviceTest -e keep true). Checks the LiteRT embedder
 * against the Python reference vectors, reproduces K4 end to end, times a search, and
 * runs a treatment turn through LayaSession. Fixtures: androidTest assets "kbtest".
 */
@RunWith(AndroidJUnit4::class)
class KbDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val test = InstrumentationRegistry.getInstrumentation().context
    private fun asset(name: String) = test.assets.open("kbtest/$name").bufferedReader().use { it.readText() }

    private fun kb(): KbBundle.Loaded {
        val kb = KbRepository.obtain(context)
        assumeTrue("install a bundle with a KB first (BundleDeviceTest -e keep true)", kb != null)
        return kb!!
    }

    @Test fun embeddingsMatchThePythonReference() {
        val m = kb().model
        val queries = JSONObject(asset("retrieval_fixture.json")).getJSONArray("queries")
        var worst = 1.0
        val t0 = System.nanoTime()
        for (i in 0 until queries.length()) {
            val q = queries.getJSONObject(i)
            val want = q.getJSONArray("vector").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
            val got = m.embed(q.getString("expanded"))
            worst = minOf(worst, want.indices.sumOf { want[it] * got[it] })
        }
        val ms = (System.nanoTime() - t0) / 1e6 / queries.length()
        println("KB_DEVICE worst cosine vs Python LiteRT reference over ${queries.length()} questions: $worst; %.1f ms per embedding".format(ms))
        assertTrue("worst cosine $worst", worst > 0.999)
    }

    @Test fun reproducesK4() {
        val kb = kb()
        val queries = asset("retrieval_queries.jsonl").lines().filter { it.isNotBlank() }.map { JSONObject(it) }
        fun relevant(c: KbChunk, e: JSONObject): Boolean {
            val words = e.getJSONArray("contains_any").let { a -> List(a.length()) { a.getString(it) } }
            return e.getString("label") in c.labels && words.any { c.text.lowercase().contains(it.lowercase()) }
        }
        val py = JSONObject(asset("results.json"))
        val pyById = py.getJSONArray("results").let { a -> (0 until a.length()).map { a.getJSONObject(it) } }.associateBy { it.getString("id") }
        var open = 0
        var filtered = 0
        val flips = mutableListOf<String>()
        val t0 = System.nanoTime()
        for (q in queries) {
            val e = q.getJSONObject("expect")
            val o = kb.retriever.search(q.getString("query"), kb.model).any { relevant(it, e) }
            val f = kb.retriever.search(q.getString("query"), kb.model, label = e.getString("label")).any { relevant(it, e) }
            if (o) open++
            if (f) filtered++
            val p = pyById.getValue(q.getString("id"))
            if (o != p.getBoolean("hybrid+names")) flips += "${q.getString("id")} open: phone $o, Python ${!o} (\"${q.getString("query")}\")"
            if (f != p.getBoolean("hybrid_filtered+names")) flips += "${q.getString("id")} filtered: phone $f, Python ${!f}"
        }
        val ms = (System.nanoTime() - t0) / 1e6 / (2 * queries.size)
        println("KB_DEVICE K4 on the phone: hybrid+names $open/${queries.size}, hybrid_filtered+names $filtered/${queries.size}; " +
            "%.1f ms per search; differs from Python (x86) on: %s".format(ms, flips.ifEmpty { listOf("none") }))
        // Same model and files, but ARM vs x86 float kernels (worst cosine ~0.9995): a near-tie in the
        // top 3 may flip. Allow one question per mode, and print it above.
        assertTrue(flips.joinToString(), flips.count { "open:" in it } <= 1 && flips.count { "filtered:" in it } <= 1)
    }

    @Test fun treatmentTurnThroughTheSession() = runBlocking {
        val kb = kb()
        val labels = context.assets.open("vision/labels_v1.json").bufferedReader().use { it.readText() }
        val templates = Templates.parse(asset("laya_templates_v1.json"))
        // Debug builds use rows still pending review (KbRepository); release builds only approved ones.
        val tools = KbTools(kb.pack.treatments, kb.retriever, kb.model, { templates.labelName(it, Lang.EN) })
        val session = LayaSession(
            ToolSchema.parse(asset("tools_schema_v1.json")), Grounding.fromLabelsJson(labels),
            { Understanding.Call("treatment", "get_treatment_info", mapOf("label" to "late_blight")) }, tools, { 0L },
            fastIntents = null,
        )
        val ctx = LayaContext("2026-10-20T09:00:00+05:30", "Nashik", true, "f1", emptyList(), emptyList(), "late_blight")
        val out = session.onUserMessage("pachheti jhulsa ka ilaj batao", ctx)
        assertTrue(out.toString(), out is TurnOutcome.Answer)
        val result = (out as TurnOutcome.Answer).results.single { it.tool == "get_treatment_info" }
        val answer = KbAnswers(kb.retriever, kb.model, templates, SafetyValidator(kb.pack.treatments, emptyList())).treatment(result, out.lang)
        println("KB_DEVICE treatment answer [${out.lang}]: ${answer.text.replace("\n", " / ")}")
        assertTrue(answer.text, answer.text.contains("pramanit vikalp") || answer.text.contains("Kripya apne Krishi"))
        assertTrue(answer.passages.all { "late_blight" in it.labels })
    }
}
