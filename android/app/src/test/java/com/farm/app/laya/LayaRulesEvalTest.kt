package com.farm.app.laya

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The rules outside the model, checked against the real Laya eval
 * (evals/laya/cases.jsonl): grounding must not ask when the expected answer is
 * a tool call, must ask about the right slot when the expected answer is a
 * clarification, and language detection must match each case's language.
 */
class LayaRulesEvalTest {
    private val repo = File("../..").canonicalFile
    private val grounding = Grounding.fromLabelsJson(File(repo, "ml/labels_v1.json").readText())
    private val cases = File(repo, "evals/laya/cases.jsonl").readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
    private val defaults = JSONObject(File(repo, "evals/laya/context_default.json").readText())

    private fun context(case: JSONObject): LayaContext {
        val c = JSONObject(defaults.toString())
        case.optJSONObject("context")?.let { o -> o.keys().forEach { c.put(it, o.get(it)) } }
        val fields = c.getJSONArray("fields").let { a -> List(a.length()) { a.getJSONObject(it) } }
        val scans = c.getJSONArray("recent_scans").let { a -> List(a.length()) { a.getJSONObject(it) } }
        return LayaContext(
            c.getString("now"), c.optString("location").takeIf { !c.isNull("location") }, c.getString("network") == "ONLINE",
            c.optString("current_field").takeIf { !c.isNull("current_field") },
            fields.map { Field(it.getString("id"), it.getString("name")) },
            scans.map { Scan(it.getString("id"), it.getString("field_id"), it.getString("label"), it.getString("taken_at")) },
            c.optJSONObject("last_diagnosis")?.getString("label"),
        )
    }

    private fun userText(case: JSONObject): String {
        val history = case.optJSONArray("history")
        val prior = if (history == null) emptyList() else List(history.length()) { history.getJSONObject(it) }
            .filter { it.getString("role") == "user" }.map { it.getString("content") }
        return (prior + case.getString("input")).joinToString(" ")
    }

    /** A correct call for an expected tool case. */
    private fun expectedCall(expect: JSONObject): Understanding.Call {
        val args = expect.getJSONObject("args")
        val m = mutableMapOf<String, Any?>()
        for (k in args.keys()) {
            val v = args.get(k)
            m[k] = when {
                v !is JSONObject -> v
                v.has("absent") -> continue
                v.has("contains_any") -> v.getJSONArray("contains_any").getString(0)
                v.has("range") -> v.getJSONArray("range").getInt(0)
                v.has("date") -> v.getString("date") + "T09:00:00+05:30"
                else -> v.getString("datetime")
            }
        }
        return Understanding.Call("x", expect.getString("tool"), m)
    }

    /** What a model guesses when a detail is missing (what L5's models did). */
    private fun guessedCall(about: String) = when (about) {
        "when" -> Understanding.Call("x", "create_reminder", mapOf("when" to "2026-10-21T08:00:00+05:30", "text" to "spray"))
        "text" -> Understanding.Call("x", "create_reminder", mapOf("when" to "2026-10-21T08:00:00+05:30", "text" to "reminder"))
        "label" -> Understanding.Call("x", "get_treatment_info", mapOf("label" to "early_blight"))
        "field" -> Understanding.Call("x", "compare_scans", mapOf("scan_a" to "s103", "scan_b" to "s101"))
        "location" -> Understanding.Call("x", "get_weather", mapOf("location" to "Nashik"))
        else -> error(about)
    }

    @Test fun groundingNeverAsksWhenTheToolCallIsRight() {
        val wrong = cases.filter { it.getJSONObject("expect").getString("action") == "tool" }
            .mapNotNull { c ->
                grounding.missing(expectedCall(c.getJSONObject("expect")), userText(c), context(c))
                    ?.let { "${c.getString("id")} (${c.getString("input")}): asked for $it" }
            }
        assertTrue(wrong.joinToString("\n"), wrong.isEmpty())
    }

    @Test fun groundingAsksForTheRightSlot() {
        val clarify = cases.filter { it.getJSONObject("expect").getString("action") == "clarify" }
        assertEquals(18, clarify.size)
        val notes = clarify.filter { it.getString("input").let { s -> "note" in s.lowercase() || "नोट" in s } }.map { it.getString("id") }.toSet()
        val wrong = clarify.mapNotNull { c ->
            val about = c.getJSONObject("expect").getString("about")
            // "write a note" cases: the guessed tool is record_observation.
            val call = if (c.getString("id") in notes) Understanding.Call("x", "record_observation", mapOf("text" to "note")) else guessedCall(about)
            val got = grounding.missing(call, userText(c), context(c))
            if (got != about) "${c.getString("id")} (${c.getString("input")}): expected $about, got $got" else null
        }
        assertTrue(wrong.joinToString("\n"), wrong.isEmpty())
    }

    @Test fun languageDetectionMatchesEvalLabels() {
        val expected = mapOf("en" to Lang.EN, "hi" to Lang.HI, "hinglish" to Lang.HINGLISH)
        val wrong = cases.filter { LanguageDetector.detect(it.getString("input")) != expected[it.getString("lang")] }
            .map { "${it.getString("id")} [${it.getString("lang")}] ${it.getString("input")} -> ${LanguageDetector.detect(it.getString("input"))}" }
        // Tolerate at most 3 of 150 (e.g. one-word inputs like "Sunday").
        assertTrue(wrong.joinToString("\n"), wrong.size <= 3)
    }
}
