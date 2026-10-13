package com.farm.app.laya

import org.json.JSONObject

/**
 * Everything Laya says, from ml/bundle/laya_templates_v1.json (L8). The SLM picks
 * *what* to do; the words come from here, in the language of the farmer's message.
 * Numbers only enter through placeholders, so the safety validator (L9) can trust
 * every digit in a rendered template.
 */
class Templates private constructor(private val root: JSONObject) {

    fun text(section: String, key: String, lang: Lang, vars: Map<String, Any> = emptyMap()): String {
        val entry = root.getJSONObject(section).optJSONObject(key)
            ?: throw IllegalArgumentException("no template $section.$key")
        return fill(entry.getString(code(lang)), vars)
    }

    fun labelName(label: String, lang: Lang): String = labelField(label, lang, "name")
    fun labelAbout(label: String, lang: Lang): String = labelField(label, lang, "about")

    private fun labelField(label: String, lang: Lang, field: String): String =
        (root.getJSONObject("labels").optJSONObject(label) ?: throw IllegalArgumentException("no label $label"))
            .getJSONObject(code(lang)).getString(field)

    /** A vision result in words: confident, uncertain (with the runner-up), or not recognised. */
    fun diagnosis(label: String, prob: Double, confident: Boolean, lang: Lang, second: String? = null): String {
        val percent = Math.round(prob * 100)
        return when {
            label == "unknown" -> text("diagnosis", "not_recognised", lang)
            label == "healthy" && confident -> text("diagnosis", "healthy", lang, mapOf("prob" to percent))
            confident -> text("diagnosis", "confident", lang, mapOf(
                "name" to labelName(label, lang), "about" to labelAbout(label, lang), "prob" to percent))
            else -> text("diagnosis", "uncertain", lang, mapOf(
                "name" to labelName(label, lang), "second" to labelName(second ?: "unknown", lang), "prob" to percent))
        }
    }

    fun clarify(slot: String, lang: Lang) = text("clarify", slot, lang)
    fun respond(kind: String, lang: Lang) = text("respond", kind, lang)

    fun fallback(reason: FallbackReason, lang: Lang) =
        text("fallback", root.getJSONObject("fallback_reasons").getString(reason.name), lang)

    fun safety(id: String, lang: Lang, vars: Map<String, Any> = emptyMap()) = text("safety", id, lang, vars)

    private fun fill(template: String, vars: Map<String, Any>): String =
        PLACEHOLDER.replace(template) { m ->
            vars[m.groupValues[1]]?.toString() ?: throw IllegalArgumentException("missing {${m.groupValues[1]}} in: $template")
        }

    companion object {
        private val PLACEHOLDER = Regex("\\{([a-z_]+)\\}")
        fun parse(json: String) = Templates(JSONObject(json))
        fun code(lang: Lang) = when (lang) { Lang.EN -> "en"; Lang.HI -> "hi"; Lang.HINGLISH -> "hinglish" }
    }
}
