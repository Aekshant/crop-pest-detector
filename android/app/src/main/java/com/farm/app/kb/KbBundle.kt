package com.farm.app.kb

import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * The KB part of a downloaded model bundle (manifest key "kb", written by
 * kb/update_manifest.py): the pack, the e5-small int8 model and its vocabulary.
 * [verify] checks size and sha256 of every file once, after download and before
 * the bundle is activated; [open] then loads them (off the main thread: the
 * model and vocabulary take a moment).
 */
class KbBundle(private val dir: File, manifestJson: String) {
    private val kb = JSONObject(manifestJson).getJSONObject("kb")
    val kbVersion: String = kb.getString("kb_version")

    fun file(key: String): File = File(dir, kb.getJSONObject("files").getJSONObject(key).getString("file"))

    /** Problems with the files on disk; empty when the bundle is intact. */
    fun verify(): List<String> {
        val files = kb.getJSONObject("files")
        return files.keys().asSequence().mapNotNull { key ->
            val spec = files.getJSONObject(key)
            val f = File(dir, spec.getString("file"))
            when {
                !f.isFile -> "$key: ${f.name} missing"
                f.length() != spec.getLong("bytes") -> "$key: ${f.name} has ${f.length()} bytes, expected ${spec.getLong("bytes")}"
                sha256(f) != spec.getString("sha256") -> "$key: ${f.name} sha256 mismatch"
                else -> null
            }
        }.toList()
    }

    /** Pack + embedder + retriever. Call only after [verify] returned no problems. */
    fun open(labelsJson: String, includeUnreviewed: Boolean = false): Loaded {
        val pack = KbPack.open(file("pack"), includeUnreviewed)
        check(pack.meta["embedding_model"] == kb.getString("embedding_model")) { "pack and manifest disagree on the embedding model" }
        val model = LiteRtEmbeddingModel(file("embedding_model"), file("embedding_table"), file("vocab"))
        return Loaded(pack, model, KbRetriever(pack.chunks, KbRetriever.namesFromLabels(labelsJson)))
    }

    class Loaded(val pack: KbPack, val model: LiteRtEmbeddingModel, val retriever: KbRetriever) : AutoCloseable {
        override fun close() = model.close()
    }

    companion object {
        /** Opens a KB from explicit files (a model bundle's kb, kb_embedder and kb_vocab artifacts). */
        fun open(pack: File, encoder: File, table: File, vocab: File, labelsJson: String, includeUnreviewed: Boolean = false): Loaded {
            val p = KbPack.open(pack, includeUnreviewed)
            val m = LiteRtEmbeddingModel(encoder, table, vocab)
            return Loaded(p, m, KbRetriever(p.chunks, KbRetriever.namesFromLabels(labelsJson)))
        }

        fun sha256(f: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            f.inputStream().use { input ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
