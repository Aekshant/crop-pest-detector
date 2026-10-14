package com.farm.app.laya

import org.json.JSONArray
import org.json.JSONObject

/**
 * One KB treatment row (design doc: the KB is the only source of products, doses and waiting periods).
 * [reviewed]: an agronomist approved it (kb/make_review_sheet.py); [rank]: their display order, 1 first.
 */
data class KbRow(
    val label: String,
    val product: String,
    val dose: String,
    val phiDays: Int,
    val registered: Boolean,
    val reviewed: Boolean = true,
    val rank: Int = Int.MAX_VALUE,
) {
    /** First word of the product, e.g. "Fixturezole" from "Fixturezole 25 EC". */
    val brand get() = product.substringBefore(' ')

    /** Laya may name it: registered for tomato and this problem, and approved. */
    val usable get() = registered && reviewed
}

/** What the vision model said last, if anything. */
data class Diagnosis(val label: String, val prob: Double, val confident: Boolean)

/** A safety reply: the templates used, the rendered text, and what the validator allowed into it. */
data class SafetyReply(val templates: List<String>, val text: String, val certainty: String?, val rows: List<KbRow>)

/**
 * L9. Questions about doses, chemicals, food safety, protective gear and
 * poisoning never reach the SLM: L4/L5 showed the models invent doses and name
 * banned chemicals (safety pass 30-37%). These rules pick a template, KB rows
 * fill its placeholders, and [SafetyValidator] checks the result.
 *
 * Returns null when the message isn't safety-sensitive (the normal flow runs).
 */
class SafetyResponder(
    private val lexicon: JSONObject,
    private val kb: List<KbRow>,
    private val templates: Templates,
    private val labelNames: Map<String, List<String>>,
) {
    private val validator = SafetyValidator(kb, list("chemicals_not_in_kb") + list("known_actives"))

    fun respond(userText: String, diagnosis: Diagnosis?, lang: Lang): SafetyReply? {
        val text = Grounding.normalize(userText)
        val products = kb.filter { text.containsWord(it.brand.lowercase()) }
        val pesticide = has(text, "pesticide_context") || products.isNotEmpty() || has(text, "chemicals_not_in_kb")
        val food = lexicon.getJSONObject("food")

        return when {
            has(text, "emergency") -> reply("T_EMERGENCY", lang)
            has(text, "ppe") -> reply("T_PPE", lang)
            has(text, "chemicals_not_in_kb") || products.any { !it.registered } ||
                (pesticide && has(text, "unregistered_cues")) -> reply("T_UNREGISTERED", lang)
            has(text, food, "cattle") -> reply("T_NO_KB_CONSULT", lang)
            pesticide && has(text, food, "children") -> reply("T_PHI_FOLLOW_LABEL", lang)
            has(text, food, "wash_or_spots") && (pesticide || has(text, food, "eat_sell_harvest")) -> reply("T_FOOD_SAFETY_GENERAL", lang)
            pesticide && has(text, food, "eat_sell_harvest") -> waitingPeriod(products, diagnosis, lang)
            pesticide && has(text, "follow_label") && !has(text, "reminder") -> reply("T_FOLLOW_LABEL", lang)
            has(text, "dose_words") || (pesticide && has(text, "quantity_words")) -> dose(text, products, diagnosis, lang)
            else -> null
        }
    }

    /** Waiting period: from the KB row of the product (or the diagnosed problem), else the label's. */
    private fun waitingPeriod(products: List<KbRow>, diagnosis: Diagnosis?, lang: Lang): SafetyReply {
        val row = products.firstOrNull { it.usable }
            ?: diagnosis?.takeIf { it.confident }?.let { d -> best(d.label) }
            ?: return reply("T_PHI_FOLLOW_LABEL", lang)
        return reply("T_PHI_FROM_KB", lang, listOf(row), mapOf("product" to row.product, "phi_days" to row.phiDays))
    }

    private fun dose(text: String, products: List<KbRow>, diagnosis: Diagnosis?, lang: Lang): SafetyReply {
        val named = labelNames.entries.firstOrNull { (_, names) -> names.any { text.contains(Grounding.normalize(it)) } }?.key
        if (named == null && diagnosis != null && !diagnosis.confident) {
            return reply("T_UNCERTAIN_RESCAN", lang, certainty = "uncertain")
        }
        val label = named ?: diagnosis?.label ?: products.firstOrNull()?.label ?: return reply("T_NEED_DIAGNOSIS", lang)
        val rows = kb.filter { it.label == label }
        val row = best(label)
        return when {
            rows.isEmpty() -> reply("T_NO_KB_CONSULT", lang)
            row == null && rows.any { it.registered } -> reply("T_NO_KB_CONSULT", lang)  // registered, but not approved yet
            row == null -> reply("T_NO_REGISTERED_PRODUCT", lang)
            else -> reply("T_KB_TREATMENT", lang, listOf(row), mapOf(
                "name" to templates.labelName(label, lang), "product" to row.product, "dose" to row.dose, "phi_days" to row.phiDays))
        }
    }

    /** The product Laya names for a problem: approved and registered, best agronomist rank first. */
    private fun best(label: String): KbRow? = kb.filter { it.label == label && it.usable }.minByOrNull { it.rank }

    private fun reply(id: String, lang: Lang, rows: List<KbRow> = emptyList(), vars: Map<String, Any> = emptyMap(),
                      certainty: String? = null): SafetyReply {
        val text = validator.enforce(templates.safety(id, lang, vars), rows, lang, templates)
        return SafetyReply(listOf(id), text, certainty, rows)
    }

    private fun list(key: String, from: JSONObject = lexicon): List<String> = from.getJSONArray(key).strings()
    private fun has(text: String, key: String) = list(key).any { text.containsWord(it) }
    private fun has(text: String, from: JSONObject, key: String) = list(key, from).any { text.containsWord(it) }

    companion object {
        fun load(lexiconJson: String, kbJson: String, templates: Templates, labelsJson: String): SafetyResponder {
            val rows = JSONObject(kbJson).getJSONArray("rows").let { a ->
                List(a.length()) { a.getJSONObject(it) }.map {
                    KbRow(it.getString("label"), it.getString("product"), it.getString("dose"), it.getInt("phi_days"),
                          it.getString("registration_status") == "registered",
                          reviewed = it.optString("review_status", "approved") == "approved",
                          rank = it.optString("display_rank").toIntOrNull() ?: Int.MAX_VALUE)
                }
            }
            val labels = JSONObject(labelsJson).getJSONArray("labels")
            val names = (0 until labels.length()).associate { i ->
                val l = labels.getJSONObject(i)
                val all = listOf(l.getString("name_en"), l.getString("name_hi")) + l.getJSONArray("aliases").strings()
                l.getString("id") to all.map { it.substringBefore('(').trim() }.filter { it.length > 2 }
            }.filterKeys { it != "unknown" && it != "healthy" }
            return SafetyResponder(JSONObject(lexiconJson), rows, templates, names)
        }
    }
}

/**
 * The last gate before any text reaches the farmer (L9): a number may appear
 * only if it came from a KB row used this turn (or the caller vouches for it,
 * e.g. the vision probability), and a product or chemical name only if it is a
 * registered KB product used this turn. [names] are the pesticide names it
 * knows besides the KB brands: banned chemicals and every active ingredient in
 * the real KB (laya_safety_v1.json, kept in sync by kb/build_kb.py), so a model
 * that names a real product from another row ("imidacloprid" when the row is
 * another product) is caught too. Sentences that break this are dropped; if
 * nothing is left, Laya says to consult the KVK.
 */
class SafetyValidator(private val kb: List<KbRow>, private val names: List<String>) {

    /** Why [text] is unsafe, or an empty list. */
    fun violations(text: String, rows: List<KbRow>, extraNumbers: Set<Double> = emptySet()): List<String> {
        val allowed = rows.filter { it.registered }
        // Every field of the row is KB text, including the strength in the product name ("25 EC").
        val numbers = allowed.flatMap { numbersIn(it.product) + numbersIn(it.dose) + numbersIn(it.phiDays.toString()) }.toSet() + extraNumbers
        val low = Grounding.normalize(text)
        val out = mutableListOf<String>()
        (numbersIn(text) - numbers).forEach { out += "number $it not from the KB" }
        // A name is fine if it is part of an allowed row's product ("Dimethomorph" in "Ametoctradin 27% + Dimethomorph 20.27% SC").
        val allowedText = allowed.joinToString(" ") { Grounding.normalize(it.product) }
        kb.map { it.brand.lowercase() }.distinct().filter { it.length > 2 && low.contains(it) && !allowedText.contains(it) }
            .forEach { out += "product $it not allowed" }
        names.map { Grounding.normalize(it) }.filter { low.containsWord(it) && !allowedText.containsWord(it) }
            .forEach { out += "chemical $it not from this turn's KB rows" }
        return out
    }

    /** [text] with unsafe sentences removed, or the consult template if none survive. */
    fun enforce(text: String, rows: List<KbRow>, lang: Lang, templates: Templates, extraNumbers: Set<Double> = emptySet()): String =
        clean(text, rows, extraNumbers) ?: templates.safety("T_NO_KB_CONSULT", lang)

    /** [text] with unsafe sentences removed, or null if none survive. */
    fun clean(text: String, rows: List<KbRow>, extraNumbers: Set<Double> = emptySet()): String? {
        if (violations(text, rows, extraNumbers).isEmpty()) return text.trim().ifEmpty { null }
        val kept = text.split(SENTENCE_END).map { it.trim() }
            .filter { it.isNotEmpty() && violations(it, rows, extraNumbers).isEmpty() }
        return if (kept.isEmpty()) null else kept.joinToString(" ")
    }

    companion object {
        // A sentence ends at . ! ? । followed by whitespace: "20.27%" and "Dr.N." stay inside their sentence.
        private val SENTENCE_END = Regex("""(?<=[.!?।])\s+""")
        private val NUMBER = Regex("""\d+(?:\.\d+)?""")

        /** Numbers as the eval scorer reads them (Devanagari digits included). */
        fun numbersIn(text: String): Set<Double> {
            val ascii = text.map { if (it in '०'..'९') '0' + (it - '०') else it }.joinToString("")
            return NUMBER.findAll(ascii).map { it.value.toDouble() }.toSet()
        }
    }
}

private fun JSONArray.strings() = List(length()) { getString(it) }

/**
 * Whole-word (or phrase) match on normalized text; a trailing * matches any
 * ending. Words are letters plus Devanagari marks, so "dho" does not match "dhoop".
 */
internal fun String.containsWord(pattern: String): Boolean {
    val p = Grounding.normalize(pattern)
    val prefix = p.endsWith("*")
    val core = Regex.escape(p.removeSuffix("*"))
    val end = if (prefix) "" else """(?![\p{L}\p{M}])"""
    return Regex("""(?<![\p{L}\p{M}])$core$end""").containsMatchIn(this)
}
