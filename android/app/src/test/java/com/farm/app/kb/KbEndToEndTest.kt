package com.farm.app.kb

import com.farm.app.laya.Grounding
import com.farm.app.laya.KbRow
import com.farm.app.laya.Lang
import com.farm.app.laya.LayaContext
import com.farm.app.laya.LayaSession
import com.farm.app.laya.SafetyValidator
import com.farm.app.laya.Templates
import com.farm.app.laya.ToolResult
import com.farm.app.laya.ToolSchema
import com.farm.app.laya.TurnOutcome
import com.farm.app.laya.Understanding
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertTrue
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

    @Test fun treatmentTurnThroughTheSessionOnTheRealPack() = runBlocking {
        val pack = File(dist, "kb_tomato_v1.sqlite")
        val model = File(dist, "e5_small_int8.onnx")
        val vocab = File(dist, "e5_small_vocab.tsv")
        assumeTrue("build kb/dist first", pack.exists() && model.exists() && vocab.exists())
        val (chunks, rows) = DriverManager.getConnection("jdbc:sqlite:${pack.path}").use { c ->
            val chunks = c.createStatement().executeQuery(
                "SELECT k.id, k.label, k.text, k.embedding, k.source_url, s.publisher FROM kb_chunk k " +
                    "JOIN (SELECT DISTINCT url, publisher FROM kb_source) s ON s.url = k.source_url ORDER BY k.id",
            ).use { r ->
                buildList { while (r.next()) add(KbChunk(r.getInt(1), KbPack.labels(r.getString(2)), r.getString(3),
                    KbRetriever.vector(r.getBytes(4)), r.getString(5), KbPack.shortPublisher(r.getString(6)))) }
            }
            // As a debug build before the agronomist review: pending rows usable, rejected never.
            val rows = c.createStatement().executeQuery(
                "SELECT label, product, dose, phi_days, registration_status, review_status, display_rank FROM kb_treatment",
            ).use { r ->
                buildList { while (r.next()) add(KbRow(r.getString(1), r.getString(2), r.getString(3), r.getInt(4),
                    r.getString(5) == "registered", reviewed = r.getString(6) != "rejected", rank = r.getString(7)?.toIntOrNull() ?: Int.MAX_VALUE)) }
            }
            chunks to rows
        }
        val labels = File(repo, "ml/labels_v1.json").readText()
        val templates = Templates.parse(File(repo, "ml/bundle/laya_templates_v1.json").readText())
        val retriever = KbRetriever(chunks, KbRetriever.namesFromLabels(labels))
        EmbeddingModel(model, vocab).use { m ->
            val tools = KbTools(rows, retriever, m, { templates.labelName(it, Lang.EN) })
            val session = LayaSession(
                ToolSchema.parse(File(repo, "ml/bundle/tools_schema_v1.json").readText()), Grounding.fromLabelsJson(labels),
                { Understanding.Call("treatment", "get_treatment_info", mapOf("label" to "late_blight")) }, tools, { 0L },
                fastIntents = null,
            )
            val ctx = LayaContext("2026-10-20T09:00:00+05:30", "Nashik", true, "f1", emptyList(), emptyList(), "late_blight")
            val out = session.onUserMessage("pachheti jhulsa ka ilaj batao", ctx)
            assertTrue(out.toString(), out is TurnOutcome.Answer)
            val result = (out as TurnOutcome.Answer).results.single { it.tool == "get_treatment_info" }
            val answer = KbAnswers(retriever, m, templates, SafetyValidator(rows, emptyList())).treatment(result, out.lang)
            println("treatment answer [${out.lang}]:\n${answer.text}")
            assertTrue(answer.text, answer.text.contains("pramanit vikalp") || answer.text.contains("Kripya apne Krishi"))
            assertTrue(answer.passages.all { "late_blight" in it.labels })
        }
    }
}
