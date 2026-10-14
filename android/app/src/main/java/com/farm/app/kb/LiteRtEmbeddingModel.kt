package com.farm.app.kb

import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

/**
 * multilingual-e5-small on LiteRT (kb/export_litert.py), the same steps as the pack
 * builder (kb/embedding.py "e5-small-litert"):
 *   "query: " + question -> UnigramTokenizer -> word vectors from the int8 table
 *   (q * row scale, memory-mapped) -> encoder.tflite (mean-pooled, L2-normalized inside).
 * Cosine to the fp32 model >= 0.999 on the KB texts. Replaces ONNX Runtime (34 MB of
 * native code); the LiteRT interpreter is already in the app for vision.
 * Not thread-safe; load once, off the main thread.
 */
class LiteRtEmbeddingModel(encoder: File, table: File, vocab: File, threads: Int = 2) : QueryEmbedder, AutoCloseable {
    private val interpreter = Interpreter(map(encoder), Interpreter.Options().setNumThreads(threads))
    private val rows: Int
    private val dim: Int
    private val scales: FloatBuffer
    private val q: ByteBuffer
    val tokenizer: UnigramTokenizer = vocab.bufferedReader().use { UnigramTokenizer.load(it) }

    init {
        val t = map(table).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { t.get(it) }
        require(magic.contentEquals("E5T1".toByteArray())) { "not an e5 table" }
        rows = t.getInt()
        dim = t.getInt()
        scales = (t.position(12) as ByteBuffer).slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().also { it.limit(rows) }
        q = (t.position(12 + 4 * rows) as ByteBuffer).slice()
        require(q.remaining() == rows * dim) { "table size mismatch" }
    }

    override fun embed(query: String): FloatArray = embedText(QUERY_PREFIX + query)

    fun embedText(text: String): FloatArray {
        val ids = tokenizer.encode(text)
        val input = Array(1) { Array(ids.size) { FloatArray(dim) } }
        for ((i, id) in ids.withIndex()) {
            require(id in 0 until rows) { "token id $id outside the table" }
            val s = scales.get(id)
            val base = id * dim
            val row = input[0][i]
            for (d in 0 until dim) row[d] = q.get(base + d) * s
        }
        interpreter.resizeInput(0, intArrayOf(1, ids.size, dim))
        interpreter.allocateTensors()
        val out = Array(1) { FloatArray(dim) }
        interpreter.run(input, out)
        return out[0]
    }

    override fun close() = interpreter.close()

    companion object {
        const val QUERY_PREFIX = "query: "

        private fun map(f: File): ByteBuffer = RandomAccessFile(f, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }
    }
}
