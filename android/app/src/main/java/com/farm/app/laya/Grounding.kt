package com.farm.app.laya

import org.json.JSONObject
import java.text.Normalizer

/**
 * "Ask when a detail is missing" (L6). The SLMs never asked a clarifying
 * question in L5 (0%); they guessed a time, a disease or a field instead. Before
 * a tool runs, each argument that must come from the farmer is checked against
 * what the farmer actually said (and the app context). If it isn't there, Laya
 * asks instead of acting on a guess.
 */
class Grounding(private val labelNames: Map<String, List<String>>) {

    /** The slot to ask about, or null when the call is grounded. */
    fun missing(call: Understanding.Call, userText: String, ctx: LayaContext): String? {
        val text = normalize(userText)
        return when (call.tool) {
            "create_reminder" -> when {
                !hasTimeWord(text) -> "when"
                contentWords(text, REMINDER_WORDS).isEmpty() -> "text"
                else -> null
            }
            "record_observation" -> if (contentWords(text, NOTE_WORDS).size < 2) "text" else null
            "get_treatment_info" -> {
                val label = call.args["label"] as? String
                val named = labelNames[label].orEmpty().any { text.contains(normalize(it)) }
                if (named || (label != null && label == ctx.lastDiagnosis)) null else "label"
            }
            "compare_scans" -> {
                val fieldNamed = ctx.fields.any { f -> f.nameWords().any { it.length > 3 && text.contains(it) } }
                if (ctx.currentField == null && !fieldNamed) "field" else null
            }
            "get_weather" -> {
                val loc = (call.args["location"] as? String).orEmpty()
                val named = loc.isNotBlank() && normalize(loc).split(" ", ",").any { it.length > 2 && text.contains(it) }
                if (ctx.location == null && !named) "location" else null
            }
            else -> null
        }
    }

    private fun hasTimeWord(text: String) =
        Regex("""\d""").containsMatchIn(text) || words(text).any { it in TIME_WORDS } ||
            TIME_PHRASES.any { text.contains(it) }

    private fun contentWords(text: String, drop: Set<String>) =
        words(text).filter { it !in drop && it !in TIME_WORDS && it !in STOP_WORDS && it.length > 1 }

    companion object {
        /** Names, Hindi names and aliases per label, from ml/labels_v1.json. */
        fun fromLabelsJson(json: String): Grounding {
            val labels = JSONObject(json).getJSONArray("labels")
            val names = (0 until labels.length()).associate { i ->
                val l = labels.getJSONObject(i)
                val aliases = l.getJSONArray("aliases").let { a -> List(a.length()) { a.getString(it) } }
                val all = listOf(l.getString("id").replace('_', ' '), l.getString("name_en"), l.getString("name_hi")) + aliases
                l.getString("id") to all.map { it.substringBefore('(').trim() }.filter { it.isNotEmpty() }
            }
            return Grounding(names)
        }

        /** Lowercase, drop the Devanagari nukta (सफ़ेद == सफेद, precomposed or not), collapse spaces. */
        fun normalize(s: String) = Normalizer.normalize(s, Normalizer.Form.NFD).lowercase().replace("़", "").replace(Regex("""\s+"""), " ").trim()

        private fun words(text: String) = Regex("""[\p{L}\p{M}]+""").findAll(text).map { it.value }.toList()

        private fun Field.nameWords() = normalize(name).split(" ").filter { it !in setOf("wala", "wale", "khet") }

        private val TIME_WORDS = setOf(
            "today", "tomorrow", "tonight", "morning", "evening", "night", "afternoon", "noon", "week", "weeks", "day",
            "days", "hour", "hours", "minutes", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
            "sunday", "am", "pm", "aaj", "kal", "parso", "subah", "shaam", "sham", "raat", "dopahar", "baje", "ghante",
            "din", "hafte", "hafta", "mahine", "somvar", "mangalvar", "budhvar", "guruvar", "shukravar", "shanivar",
            "ravivar", "आज", "कल", "परसों", "सुबह", "शाम", "रात", "दोपहर", "बजे", "घंटे", "दिन", "हफ्ते", "हफ्ता", "महीने",
            "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार",
        )
        private val TIME_PHRASES = listOf("o'clock", "next week")
        private val REMINDER_WORDS = setOf(
            "remind", "reminder", "set", "me", "yaad", "dilana", "dilao", "dilaana", "dilaiye", "laga", "lagao", "do",
            "याद", "दिलाना", "दिलाओ", "दिलाइए", "रिमाइंडर", "लगा", "लगाओ", "दो",
        )
        private val NOTE_WORDS = setOf(
            "note", "write", "record", "log", "add", "down", "likh", "likho", "likhna", "lo", "karo", "kar", "note",
            "नोट", "लिखो", "लिख", "लिखना", "लो", "करो", "कर", "दर्ज",
        )
        private val STOP_WORDS = setOf(
            "a", "an", "the", "to", "for", "of", "that", "i", "please", "my", "it", "this", "is", "ka", "ki", "ke", "ko",
            "me", "mein", "hai", "ek", "na", "bhi", "liye", "की", "का", "के", "को", "में", "है", "एक", "भी", "लिए",
        )
    }
}
