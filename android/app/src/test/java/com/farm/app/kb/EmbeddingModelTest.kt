package com.farm.app.kb

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The phone's tokenizer and embedder match the reference (kb/embedding.py) on
 * the real model. Needs kb/dist/ from kb/export_embedder.py, kb/build_pack.py,
 * evals/kb/eval_retrieval.py --fixture and kb/tokenizer_fixture.py; skipped otherwise.
 */
class EmbeddingModelTest {
    private val dist = File(File("../..").canonicalFile, "kb/dist")
    private val vocab = File(dist, "e5_small_vocab.tsv")
    private val model = File(dist, "e5_small_int8.onnx")

    @Test fun tokenizerMatchesReference() {
        val fixture = File(dist, "tokenizer_fixture.json")
        assumeTrue("export the embedder and tokenizer fixture first", vocab.exists() && fixture.exists())
        val tok = vocab.bufferedReader().use { UnigramTokenizer.load(it) }
        val cases = JSONArray(fixture.readText())
        val wrong = mutableListOf<String>()
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val want = c.getJSONArray("ids").let { a -> List(a.length()) { a.getInt(it) } }
            val got = tok.encode(c.getString("text")).toList()
            if (got != want) wrong += "${c.getString("text").take(60)}\n   want $want\n   got  $got"
        }
        assertTrue("${wrong.size}/${cases.length()} differ:\n" + wrong.take(5).joinToString("\n"), wrong.isEmpty())
    }

    @Test fun embeddingsMatchReference() {
        val fixture = File(dist, "retrieval_fixture.json")
        assumeTrue("export the embedder and retrieval fixture first", model.exists() && vocab.exists() && fixture.exists())
        EmbeddingModel(model, vocab).use { m ->
            val queries = JSONObject(fixture.readText()).getJSONArray("queries")
            var worst = 1.0
            for (i in 0 until queries.length()) {
                val q = queries.getJSONObject(i)
                val want = q.getJSONArray("vector").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
                val got = m.embed(q.getString("expanded"))
                worst = minOf(worst, want.indices.sumOf { want[it] * got[it] })
            }
            println("worst cosine vs reference over ${queries.length()} questions: $worst")
            assertTrue("worst cosine $worst", worst > 0.9999)
        }
    }

    @Test fun shortQuestionTokens() {
        assumeTrue(vocab.exists())
        val tok = vocab.bufferedReader().use { UnigramTokenizer.load(it) }
        val ids = tok.encode("query: hello")
        assertEquals(UnigramTokenizer.BOS, ids.first())
        assertEquals(UnigramTokenizer.EOS, ids.last())
    }
}
