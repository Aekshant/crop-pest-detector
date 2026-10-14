package com.farm.app.kb

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * multilingual-e5-small (int8 ONNX, kb/export_embedder.py) on ONNX Runtime:
 * "query: " + question -> tokens -> last_hidden_state -> mean over tokens -> unit vector.
 * The same steps kb/embedding.py uses to build the pack, so a question lands in
 * the same space as the passages. Not thread-safe; load once, off the main thread.
 */
class EmbeddingModel(model: File, vocab: File) : QueryEmbedder, AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(model.path, OrtSession.SessionOptions())
    val tokenizer: UnigramTokenizer = vocab.bufferedReader().use { UnigramTokenizer.load(it) }

    override fun embed(query: String): FloatArray = embedText(QUERY_PREFIX + query)

    fun embedText(text: String): FloatArray {
        val ids = tokenizer.encode(text).map { it.toLong() }.toLongArray()
        val shape = longArrayOf(1, ids.size.toLong())
        fun tensor(values: LongArray) = OnnxTensor.createTensor(env, LongBuffer.wrap(values), shape)
        tensor(ids).use { input ->
            tensor(LongArray(ids.size) { 1 }).use { mask ->
                tensor(LongArray(ids.size)).use { types ->
                    session.run(mapOf("input_ids" to input, "attention_mask" to mask, "token_type_ids" to types)).use { out ->
                        @Suppress("UNCHECKED_CAST")
                        val hidden = (out[0].value as Array<Array<FloatArray>>)[0]
                        val v = FloatArray(hidden[0].size)
                        for (tok in hidden) for (d in v.indices) v[d] += tok[d]
                        val norm = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
                        for (d in v.indices) v[d] /= norm  // the mean's 1/n cancels in the normalization
                        return v
                    }
                }
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val QUERY_PREFIX = "query: "
    }
}
