package com.farm.app.kb

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.sql.DriverManager

/**
 * K4 on the phone's code path: question -> UnigramTokenizer -> e5-small int8 on
 * ONNX Runtime -> KbRetriever, scored with the same relevance rule as
 * evals/kb/eval_retrieval.py. Must reproduce the Python hit rates.
 */
class KbEndToEndTest {
    private val repo = File("../..").canonicalFile
    private val dist = File(repo, "kb/dist")

    @Test fun phonePipelineReproducesK4() {
        val pack = File(dist, "kb_tomato_v1.sqlite")
        val model = File(dist, "e5_small_int8.onnx")
        val vocab = File(dist, "e5_small_vocab.tsv")
        val results = File(repo, "evals/kb/results_e5-small-int8.json")
        assumeTrue("build kb/dist first", pack.exists() && model.exists() && vocab.exists() && results.exists())

        val chunks = DriverManager.getConnection("jdbc:sqlite:${pack.path}").use { c ->
            c.createStatement().executeQuery("SELECT id, label, text, embedding FROM kb_chunk ORDER BY id").use { r ->
                buildList { while (r.next()) add(KbChunk(r.getInt(1), KbPack.labels(r.getString(2)), r.getString(3), KbRetriever.vector(r.getBytes(4)))) }
            }
        }
        val retriever = KbRetriever(chunks, KbRetriever.namesFromLabels(File(repo, "ml/labels_v1.json").readText()))
        val queries = File(repo, "evals/kb/retrieval_queries.jsonl").readLines().filter { it.isNotBlank() }.map { JSONObject(it) }

        fun relevant(chunk: KbChunk, expect: JSONObject): Boolean {
            val words = expect.getJSONArray("contains_any").let { a -> List(a.length()) { a.getString(it) } }
            return expect.getString("label") in chunk.labels && words.any { chunk.text.lowercase().contains(it.lowercase()) }
        }

        var open = 0
        var filtered = 0
        val t0 = System.nanoTime()
        EmbeddingModel(model, vocab).use { m ->
            for (q in queries) {
                val expect = q.getJSONObject("expect")
                if (retriever.search(q.getString("query"), m).any { relevant(it, expect) }) open++
                if (retriever.search(q.getString("query"), m, label = expect.getString("label")).any { relevant(it, expect) }) filtered++
            }
        }
        val ms = (System.nanoTime() - t0) / 1e6 / (2 * queries.size)
        println("phone pipeline: hybrid+names $open/${queries.size}, hybrid_filtered+names $filtered/${queries.size}, %.1f ms per search (JVM, x86)".format(ms))

        val py = JSONObject(results.readText())
        assertEquals(Math.round(py.getJSONObject("hybrid+names").getDouble("hit_rate") * queries.size).toInt(), open)
        assertEquals(Math.round(py.getJSONObject("hybrid_filtered+names").getDouble("hit_rate") * queries.size).toInt(), filtered)
    }
}
