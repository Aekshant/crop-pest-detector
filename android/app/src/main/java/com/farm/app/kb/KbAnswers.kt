package com.farm.app.kb

import com.farm.app.laya.KbRow
import com.farm.app.laya.Lang
import com.farm.app.laya.LanguageDetector
import com.farm.app.laya.LayaContext
import com.farm.app.laya.SafetyValidator
import com.farm.app.laya.Templates
import com.farm.app.laya.ToolResult
import com.farm.app.laya.ToolRunner
import com.farm.app.laya.Understanding

/**
 * get_treatment_info (tools_schema_v1: "Verified treatment and prevention for a
 * diagnosed problem, from the knowledge base"). Data: "label", "rows" (approved,
 * registered KB rows, best rank first) and "passages" (top prevention passages
 * for that label). Other tools go to [next].
 */
class KbTools(
    private val rows: List<KbRow>,
    private val retriever: KbRetriever,
    private val embedder: QueryEmbedder,
    private val englishName: (String) -> String,
    private val next: ToolRunner? = null,
) : ToolRunner {
    override suspend fun run(call: Understanding.Call, context: LayaContext): ToolResult {
        if (call.tool != TOOL) return next?.run(call, context) ?: ToolResult.Failed("no runner for ${call.tool}")
        val label = call.args["label"] as? String ?: return ToolResult.Failed("label missing")
        val usable = rows.filter { it.label == label && it.usable }.sortedBy { it.rank }
        val passages = retriever.search(preventionQuery(englishName(label)), embedder, label = label)
        return ToolResult.Ok(TOOL, mapOf("label" to label, "rows" to usable, "passages" to passages))
    }

    companion object {
        const val TOOL = "get_treatment_info"
        fun preventionQuery(name: String) = "How to prevent and manage $name in tomato: cultural practices, field sanitation, resistant varieties"
    }
}

/** The SLM (edge or cloud) writing a short explanation from retrieved passages. Cancellable; may throw. */
fun interface ExplanationModel {
    suspend fun complete(system: String, user: String): String
}

/** A KB answer: the text to show, the passages it rests on, and whether a model wrote it. */
data class KbAnswer(val text: String, val passages: List<KbChunk>, val generated: Boolean)

/**
 * The words for KB answers (design doc: "Diagnosis explanation (why?, kya karu?) -> Laya,
 * with the diagnosis and retrieved KB chunks in context; SLM: explanation only").
 *
 *  - explain(): passages for the question (label-filtered) -> the model explains in the
 *    farmer's language from those passages only -> SafetyValidator (no numbers, no product
 *    names) -> language check. Any failure falls back to the label's template description
 *    plus the best passage verbatim, attributed to its source.
 *  - treatment(): T_KB_TREATMENT for the best approved row (or the consult template) and
 *    the best prevention passage, attributed.
 */
class KbAnswers(
    private val retriever: KbRetriever,
    private val embedder: QueryEmbedder,
    private val templates: Templates,
    private val validator: SafetyValidator,
    private val model: ExplanationModel? = null,
) {
    suspend fun explain(question: String, label: String, lang: Lang): KbAnswer {
        val passages = retriever.search(question, embedder, label = label)
        val generated = model?.let { m ->
            runCatching { m.complete(systemPrompt(lang), userPrompt(question, label, passages)) }.getOrNull()
                ?.let { validator.clean(it.take(MAX_CHARS), emptyList()) }
                ?.takeIf { passages.isNotEmpty() && LanguageDetector.detect(it) == lang }
        }
        if (generated != null) return KbAnswer(generated, passages, generated = true)
        val about = templates.labelAbout(label, lang)
        return KbAnswer(listOfNotNull(about, sourced(passages, lang)).joinToString(" "), passages, generated = false)
    }

    fun treatment(result: ToolResult.Ok, lang: Lang): KbAnswer {
        @Suppress("UNCHECKED_CAST") val rows = result.data["rows"] as List<KbRow>
        @Suppress("UNCHECKED_CAST") val passages = result.data["passages"] as List<KbChunk>
        val label = result.data["label"] as String
        val row = rows.firstOrNull()
        val treatment = if (row == null) templates.safety("T_NO_KB_CONSULT", lang) else validator.enforce(
            templates.safety("T_KB_TREATMENT", lang, mapOf("name" to templates.labelName(label, lang), "product" to row.product,
                                                           "dose" to row.dose, "phi_days" to row.phiDays)),
            listOf(row), lang, templates,
        )
        val prevention = sourced(passages, lang)?.let { "${templates.text("kb", "prevention_heading", lang)} $it" }
        return KbAnswer(listOfNotNull(treatment, prevention).joinToString("\n"), passages, generated = false)
    }

    /** The best passage that survives the validator, verbatim, with its source. */
    private fun sourced(passages: List<KbChunk>, lang: Lang): String? {
        for (p in passages) {
            val clean = validator.clean(p.text, emptyList()) ?: continue
            return templates.text("kb", "from_source", lang, mapOf("source" to p.source.ifEmpty { "KB" }, "passage" to clean))
        }
        return null
    }

    companion object {
        const val MAX_CHARS = 400

        fun systemPrompt(lang: Lang): String {
            val language = when (lang) {
                Lang.EN -> "English"
                Lang.HI -> "Hindi in Devanagari script"
                Lang.HINGLISH -> "Hinglish (Hindi in Roman letters)"
            }
            return """You are Laya, a tomato farming assistant. Answer the farmer's question in $language, in 2 or 3 short sentences.
Use ONLY the numbered passages. If they do not answer the question, say you don't have verified information.
Never mention any pesticide, product, chemical, dose, quantity or number. Do not add advice that is not in the passages."""
        }

        fun userPrompt(question: String, label: String, passages: List<KbChunk>): String =
            "Diagnosis: ${label.replace('_', ' ')}\n\nPassages:\n" +
                passages.mapIndexed { i, p -> "[${i + 1}] ${p.text}" }.joinToString("\n") +
                "\n\nFarmer's question: $question"
    }
}
