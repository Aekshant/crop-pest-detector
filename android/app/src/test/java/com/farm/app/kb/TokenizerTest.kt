package com.farm.app.kb

import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The phone's tokenizer gives the reference ids (HF tokenizers) on every text the KB pack
 * embeds. Needs kb/dist/ from kb/export_litert.py, kb/build_pack.py,
 * evals/kb/eval_retrieval.py --fixture and kb/tokenizer_fixture.py; skipped otherwise.
 * (Embedding parity runs on the phone: KbDeviceTest. LiteRT has no desktop JVM build.)
 */
class TokenizerTest {
    private val dist = File(File("../..").canonicalFile, "kb/dist")
    private val vocab = File(dist, "e5_small_vocab.tsv")

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

    @Test fun shortQuestionTokens() {
        assumeTrue(vocab.exists())
        val tok = vocab.bufferedReader().use { UnigramTokenizer.load(it) }
        val ids = tok.encode("query: hello")
        assertEquals(UnigramTokenizer.BOS, ids.first())
        assertEquals(UnigramTokenizer.EOS, ids.last())
    }
}
