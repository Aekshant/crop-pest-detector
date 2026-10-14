package com.farm.app.kb

import java.io.BufferedReader
import java.text.Normalizer

/**
 * The XLM-R / multilingual-e5 tokenizer (SentencePiece Unigram), ported from
 * HF tokenizers so the phone needs no native tokenizer library:
 *
 *  1. normalize: NFKC (the SentencePiece "nmt_nfkc" charsmap, which also maps
 *     unusual whitespace to a space)
 *  2. Metaspace: spaces become "▁", a "▁" is prepended, and the text is split
 *     before each "▁"
 *  3. Unigram: Viterbi over the vocabulary's log-probabilities per word;
 *     characters no piece covers become <unk> (consecutive ones fused)
 *  4. <s> ... </s>
 *
 * EmbeddingModelTest checks the ids against the reference tokenizer on every
 * text the KB pack embeds.
 */
class UnigramTokenizer(pieces: List<String>, private val scores: FloatArray) {
    private val ids = HashMap<String, Int>(pieces.size * 2).apply { pieces.forEachIndexed { i, p -> putIfAbsent(p, i) } }
    private val maxLen = pieces.maxOf { it.length }
    private val unkScore = scores.filterIndexed { i, _ -> i > UNK }.minOrNull()!! - 10f

    fun encode(text: String, maxTokens: Int = 512): IntArray {
        val out = ArrayList<Int>()
        out += BOS
        for (word in words(normalize(text))) out += viterbi(word)
        if (out.size > maxTokens - 1) out.subList(maxTokens - 1, out.size).clear()
        out += EOS
        return out.toIntArray()
    }

    private fun normalize(s: String): String {
        val n = Normalizer.normalize(s, Normalizer.Form.NFKC)
        // SentencePiece also drops leading whitespace and collapses runs to one space (a trailing one stays).
        return buildString(n.length) {
            for (c in n) when {
                c == '\t' || c == '\n' || c == '\r' || Character.isSpaceChar(c) -> if (isNotEmpty() && last() != ' ') append(' ')
                Character.getType(c) == Character.CONTROL.toInt() -> {}
                else -> append(c)
            }
        }
    }

    private fun words(s: String): List<String> {
        if (s.isEmpty()) return emptyList()
        val meta = s.replace(" ", SPACE).let { if (it.startsWith(SPACE)) it else SPACE + it }
        val out = ArrayList<String>()
        var start = 0
        for (i in 1 until meta.length) if (meta[i] == SPACE[0]) { out += meta.substring(start, i); start = i }
        out += meta.substring(start)
        return out
    }

    private fun viterbi(w: String): List<Int> {
        val n = w.length
        val best = FloatArray(n + 1) { Float.NEGATIVE_INFINITY }.also { it[0] = 0f }
        val from = IntArray(n + 1)
        val piece = IntArray(n + 1)
        for (i in 0 until n) {
            if (best[i] == Float.NEGATIVE_INFINITY) continue
            var matched = false
            val charLen = if (Character.isHighSurrogate(w[i]) && i + 1 < n) 2 else 1
            for (len in 1..minOf(maxLen, n - i)) {
                val id = ids[w.substring(i, i + len)] ?: continue
                if (len >= charLen) matched = matched || len == charLen
                val sc = best[i] + scores[id]
                if (sc > best[i + len]) { best[i + len] = sc; from[i + len] = i; piece[i + len] = id }
            }
            if (!matched) {
                val sc = best[i] + unkScore
                if (sc > best[i + charLen]) { best[i + charLen] = sc; from[i + charLen] = i; piece[i + charLen] = UNK }
            }
        }
        val out = ArrayList<Int>()
        var j = n
        while (j > 0) { out += piece[j]; j = from[j] }
        out.reverse()
        // Fuse consecutive <unk> (HF Unigram fuse_unk).
        return out.filterIndexed { i, id -> !(id == UNK && i > 0 && out[i - 1] == UNK) }
    }

    companion object {
        const val BOS = 0
        const val EOS = 2
        const val UNK = 3
        private const val SPACE = "▁"

        /** kb/dist/e5_small_vocab.tsv: line i = "piece<TAB>score" of id i. */
        fun load(reader: BufferedReader): UnigramTokenizer {
            val pieces = ArrayList<String>(250_002)
            val scores = ArrayList<Float>(250_002)
            reader.forEachLine { line ->
                val tab = line.lastIndexOf('\t')
                pieces += line.substring(0, tab)
                scores += line.substring(tab + 1).toFloat()
            }
            return UnigramTokenizer(pieces, scores.toFloatArray())
        }
    }
}
