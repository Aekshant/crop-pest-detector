package com.farm.app.laya

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class TemplatesTest {
    private val repo = File("../..").canonicalFile
    private val t = Templates.parse(File(repo, "ml/bundle/laya_templates_v1.json").readText())
    private val labels = JSONObject(File(repo, "ml/labels_v1.json").readText()).getJSONArray("labels")
        .let { a -> List(a.length()) { a.getJSONObject(it).getString("id") } }

    @Test fun everyLabelRendersInEveryLanguage() {
        for (label in labels) for (lang in Lang.values()) {
            for (confident in listOf(true, false)) {
                val s = t.diagnosis(label, 0.83, confident, lang, second = "healthy")
                assertFalse("$label $lang: $s", s.contains("{"))
                if (label != "unknown") assertTrue("$label $lang: $s", s.contains("83%"))
            }
        }
    }

    @Test fun replyFollowsLanguage() {
        assertTrue(t.diagnosis("early_blight", 0.9, true, Lang.EN).startsWith("This looks like Early blight"))
        assertTrue(t.diagnosis("early_blight", 0.9, true, Lang.HI).startsWith("यह अगेती झुलसा"))
        assertTrue(t.diagnosis("early_blight", 0.9, true, Lang.HINGLISH).startsWith("Ye Agati jhulsa"))
    }

    @Test fun everyStateMachineOutcomeHasWords() {
        for (lang in Lang.values()) {
            listOf("when", "text", "label", "field", "location", "photo").forEach { assertTrue(t.clarify(it, lang).isNotBlank()) }
            FallbackReason.values().forEach { assertTrue(t.fallback(it, lang).isNotBlank()) }
            listOf("greeting", "thanks", "out_of_scope", "offline", "unsupported").forEach { assertTrue(t.respond(it, lang).isNotBlank()) }
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun missingPlaceholderIsAnError() {
        t.safety("T_KB_TREATMENT", Lang.EN, mapOf("name" to "Early blight"))
    }
}
