package com.farm.app.kb

import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer
import kotlin.math.ln

/** One advisory passage from the KB pack (kb_chunk), with its unit embedding. */
class KbChunk(val id: Int, val labels: Set<String>, val text: String, val vector: FloatArray)

/** Embeds a question with the pack's model (meta.embedding_model), including its query prefix, as a unit vector. */
fun interface QueryEmbedder {
    fun embed(query: String): FloatArray
}

/**
 * K3/K4 retrieval on the phone, the same pipeline evals/kb/eval_retrieval.py
 * scores as "hybrid(_filtered)+names" (88% top-3 with a diagnosis, 82% without):
 *
 *  1. local names -> English: "agati jhulsa" gets " (Early blight)" appended
 *  2. vector ranking: cosine against every chunk (brute force: ~200 x 384 floats)
 *  3. BM25 keyword ranking on the expanded question
 *  4. reciprocal rank fusion of 2 and 3
 *  5. if the diagnosis is known, only chunks of that label
 *
 * Chunks carry no doses (removed at build time); doses come from kb_treatment.
 */
class KbRetriever(private val chunks: List<KbChunk>, private val names: List<Pair<String, String>>) {
    private val bm25 = Bm25(chunks.map { it.text })

    fun expand(query: String): String {
        val q = norm(query)
        val found = names.filter { (local, _) -> local.isNotEmpty() && q.contains(local) }.map { it.second }.toSortedSet()
        return if (found.isEmpty()) query else "$query (${found.joinToString(", ")})"
    }

    fun search(query: String, embedder: QueryEmbedder, label: String? = null, k: Int = 3): List<KbChunk> {
        val expanded = expand(query)
        val v = embedder.embed(expanded)
        val byVector = chunks.indices.sortedByDescending { dot(v, chunks[it].vector) }
        val fused = rrf(byVector, bm25.rank(expanded))
        return fused.asSequence().map { chunks[it] }.filter { label == null || label in it.labels }.take(k).toList()
    }

    private fun dot(a: FloatArray, b: FloatArray): Double {
        var s = 0.0
        for (i in a.indices) s += a[i].toDouble() * b[i]
        return s
    }

    companion object {
        private const val RRF_K = 60

        /** Lowercase, NFD, no Devanagari nukta: the eval's norm(). */
        fun norm(s: String): String = Normalizer.normalize(s, Normalizer.Form.NFD).lowercase().replace("़", "")

        fun rrf(vararg rankings: List<Int>): List<Int> {
            val score = LinkedHashMap<Int, Double>()
            for (ranking in rankings) ranking.forEachIndexed { r, i -> score[i] = (score[i] ?: 0.0) + 1.0 / (RRF_K + r + 1) }
            return score.entries.sortedByDescending { it.value }.map { it.key }
        }

        /** (local name, English name) pairs from ml/labels_v1.json: Hindi names and aliases. */
        fun namesFromLabels(labelsJson: String): List<Pair<String, String>> {
            val labels = JSONObject(labelsJson).getJSONArray("labels")
            val out = mutableListOf<Pair<String, String>>()
            for (i in 0 until labels.length()) {
                val l = labels.getJSONObject(i)
                if (l.getString("id") in setOf("unknown", "healthy")) continue
                val en = l.getString("name_en").substringBefore('(').trim()
                val aliases = l.getJSONArray("aliases")
                val local = listOf(l.getString("name_hi")) + List(aliases.length()) { aliases.getString(it) }
                local.forEach { out += norm(it.substringBefore('(').trim()) to en }
            }
            return out
        }

        /** kb_chunk.embedding: float32 little-endian. */
        fun vector(blob: ByteArray): FloatArray {
            val fb = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(fb.remaining()).also { fb.get(it) }
        }
    }
}

/** Okapi BM25 (k1 = 1.2, b = 0.75) over the chunk texts. */
internal class Bm25(docs: List<String>, private val k1: Double = 1.2, private val b: Double = 0.75) {
    private val tf = docs.map { d -> tokens(d).groupingBy { it }.eachCount() }
    private val len = tf.map { it.values.sum() }
    private val avg = len.average()
    private val idf: Map<String, Double> = run {
        val df = HashMap<String, Int>()
        tf.forEach { t -> t.keys.forEach { df[it] = (df[it] ?: 0) + 1 } }
        df.mapValues { (_, d) -> ln(1 + (docs.size - d + 0.5) / (d + 0.5)) }
    }

    fun rank(query: String): List<Int> {
        val q = tokens(query).toSet()
        return tf.indices.map { i ->
            i to q.sumOf { w ->
                val f = tf[i][w] ?: return@sumOf 0.0
                idf.getValue(w) * f * (k1 + 1) / (f + k1 * (1 - b + b * len[i] / avg))
            }
        }.sortedByDescending { it.second }.filter { it.second > 0 }.map { it.first }
    }

    companion object {
        private val TOKEN = Regex("[\\p{L}\\p{N}_\\u0900-\\u097F]+")
        fun tokens(s: String) = TOKEN.findAll(KbRetriever.norm(s)).map { it.value }.toList()
    }
}
