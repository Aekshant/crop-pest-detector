package com.farm.app.kb

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.sql.DriverManager

/**
 * The Kotlin retriever ranks exactly like evals/kb/eval_retrieval.py on the real
 * pack. Needs kb/dist/ from:
 *   uvx --with fastembed --with sqlite-vec --with pdfplumber python kb/build_pack.py
 *   uvx --with fastembed --with sqlite-vec python evals/kb/eval_retrieval.py --fixture kb/dist/retrieval_fixture.json
 * (skipped when they are missing, e.g. in CI).
 */
class KbRetrieverTest {
    private val repo = File("../..").canonicalFile
    private val pack = File(repo, "kb/dist/kb_tomato_v1.sqlite")
    private val fixture = File(repo, "kb/dist/retrieval_fixture.json")
    private val names = KbRetriever.namesFromLabels(File(repo, "ml/labels_v1.json").readText())

    private fun chunks(): List<KbChunk> = DriverManager.getConnection("jdbc:sqlite:${pack.path}").use { c ->
        c.createStatement().executeQuery("SELECT id, label, text, embedding FROM kb_chunk ORDER BY id").use { r ->
            buildList { while (r.next()) add(KbChunk(r.getInt(1), KbPack.labels(r.getString(2)), r.getString(3), KbRetriever.vector(r.getBytes(4)))) }
        }
    }

    @Test fun sameRankingAsThePythonEval() {
        assumeTrue("build the pack and fixture first", pack.exists() && fixture.exists())
        val retriever = KbRetriever(chunks(), names)
        val queries = JSONObject(fixture.readText()).getJSONArray("queries")
        var same = 0
        for (i in 0 until queries.length()) {
            val q = queries.getJSONObject(i)
            assertEquals(q.getString("id"), q.getString("expanded"), retriever.expand(q.getString("query")))
            val vec = q.getJSONArray("vector").let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
            val embedder = QueryEmbedder { vec }
            fun ids(key: String) = q.getJSONArray(key).let { a -> List(a.length()) { a.getInt(it) } }
            val open = retriever.search(q.getString("query"), embedder).map { it.id }
            val filtered = retriever.search(q.getString("query"), embedder, label = q.getString("label")).map { it.id }
            if (open == ids("hybrid_ids") && filtered == ids("hybrid_filtered_ids")) same++
            else println("differs ${q.getString("id")}: $open/${ids("hybrid_ids")}  $filtered/${ids("hybrid_filtered_ids")}")
        }
        assertEquals("queries ranked identically to Python", queries.length(), same)
    }

    @Test fun localNamesExpandToEnglish() {
        val r = KbRetriever(emptyList(), names)
        assertEquals("agati jhulsa ke dhabbe (Early blight)", r.expand("agati jhulsa ke dhabbe"))
        assertEquals("लाल मकड़ी से नुकसान (Spider mites)", r.expand("लाल मकड़ी से नुकसान"))
        assertEquals("how do I water tomatoes", r.expand("how do I water tomatoes"))
    }

    @Test fun rrfFusesRankings() {
        assertEquals(listOf(2, 1, 3), KbRetriever.rrf(listOf(1, 2), listOf(2, 3)))
        assertTrue(Bm25(listOf("whitefly vector of leaf curl", "fruit borer holes")).rank("whitefly") == listOf(0))
    }
}
