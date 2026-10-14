package com.farm.app.kb

import com.farm.app.laya.KbRow
import com.farm.app.laya.Lang
import com.farm.app.laya.LayaContext
import com.farm.app.laya.SafetyValidator
import com.farm.app.laya.Templates
import com.farm.app.laya.ToolResult
import com.farm.app.laya.Understanding
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class KbAnswersTest {
    private val repo = File("../..").canonicalFile
    private val templates = Templates.parse(File(repo, "ml/bundle/laya_templates_v1.json").readText())

    // Two passages on early blight, one on whitefly; a fake embedder that maps every question near passage 1.
    private fun v(vararg x: Float) = x
    private val chunks = listOf(
        KbChunk(1, setOf("early_blight"), "Early blight, management: Remove and destroy affected leaves. Use resistant varieties.",
                v(1f, 0f, 0f), "https://x/tnau", "TNAU Agritech Portal"),
        KbChunk(2, setOf("early_blight"), "Early blight, favourable conditions: Warm, humid (24-29 degree C), rainy weather.",
                v(0.9f, 0.43f, 0f), "https://x/tnau", "TNAU Agritech Portal"),
        KbChunk(3, setOf("whitefly"), "Whitefly, management: Remove alternate weed host Abutilon indicum.", v(0f, 0f, 1f)),
    )
    private val retriever = KbRetriever(chunks, emptyList())
    private val embedder = QueryEmbedder { v(1f, 0f, 0f) }
    private val validator = SafetyValidator(emptyList(), listOf("mancozeb", "monocrotophos"))
    private fun answers(model: ExplanationModel?) = KbAnswers(retriever, embedder, templates, validator, model)

    @Test fun explanationFromTheModelIsUsedWhenSafeAndInTheRightLanguage() = runBlocking {
        val a = answers { _, _ -> "प्रभावित पत्तियाँ तोड़कर नष्ट करें। प्रतिरोधी किस्में लगाएँ।" }.explain("kya karu", "early_blight", Lang.HI)
        assertTrue(a.generated)
        assertEquals("प्रभावित पत्तियाँ तोड़कर नष्ट करें। प्रतिरोधी किस्में लगाएँ।", a.text)
        assertEquals(listOf(1, 2), a.passages.map { it.id })
    }

    @Test fun unsafeSentencesAreDroppedFromTheModelText() = runBlocking {
        val a = answers { _, _ -> "Remove affected leaves. Spray mancozeb at 2 g per litre." }.explain("what to do", "early_blight", Lang.EN)
        assertEquals("Remove affected leaves.", a.text)
    }

    @Test fun wrongLanguageOrFailureFallsBackToTemplateAndSource() = runBlocking {
        val english = answers { _, _ -> "Remove affected leaves." }.explain("kya karu", "early_blight", Lang.HI)
        assertFalse(english.generated)
        assertTrue(english.text, english.text.startsWith(templates.labelAbout("early_blight", Lang.HI)))
        assertTrue(english.text, english.text.contains("TNAU Agritech Portal के अनुसार"))
        val failing = answers { _, _ -> throw IllegalStateException("model down") }.explain("why", "early_blight", Lang.EN)
        assertTrue(failing.text, failing.text.contains("From TNAU Agritech Portal: Early blight, management: Remove and destroy"))
        assertFalse(answers(null).explain("why", "early_blight", Lang.EN).generated)
    }

    @Test fun passagesWithNumbersAreNotShownVerbatim() = runBlocking {
        // Passage 2 has "24-29": only passage 1 may be quoted.
        val a = answers(null).explain("weather", "early_blight", Lang.EN)
        assertFalse(a.text, a.text.contains("24"))
    }

    @Test fun promptCarriesPassagesLanguageAndRules() {
        assertTrue(KbAnswers.systemPrompt(Lang.HINGLISH).contains("Hinglish"))
        assertTrue(KbAnswers.systemPrompt(Lang.EN).contains("Never mention any pesticide"))
        val u = KbAnswers.userPrompt("why?", "early_blight", chunks.take(2))
        assertTrue(u.contains("[1] Early blight, management") && u.contains("[2]") && u.contains("Diagnosis: early blight"))
    }

    @Test fun treatmentUsesOnlyApprovedRowsBestRankFirst() = runBlocking {
        val rows = listOf(
            KbRow("early_blight", "Alpha 50% WP", "500 g per hectare in 500 litres of water", 7, true, reviewed = true, rank = 2),
            KbRow("early_blight", "Beta 25% SC", "300 ml per hectare in 500 litres of water", 3, true, reviewed = true, rank = 1),
            KbRow("early_blight", "Gamma 10% EC", "100 ml per hectare in 500 litres of water", 1, true, reviewed = false),
        )
        val tools = KbTools(rows, retriever, embedder, { "Early blight" })
        val ctx = LayaContext("2026-10-20T09:00:00+05:30", null, true, null, emptyList(), emptyList(), "early_blight")
        val result = tools.run(Understanding.Call("t", "get_treatment_info", mapOf("label" to "early_blight")), ctx) as ToolResult.Ok
        @Suppress("UNCHECKED_CAST") assertEquals(listOf("Beta 25% SC", "Alpha 50% WP"), (result.data["rows"] as List<KbRow>).map { it.product })
        val text = answers(null).treatment(result, Lang.EN).text
        assertTrue(text, text.startsWith("For Early blight, the verified option is Beta 25% SC: 300 ml per hectare in 500 litres of water."))
        assertTrue(text, text.contains("To prevent it: From TNAU Agritech Portal:"))
    }

    @Test fun noApprovedRowMeansConsult() = runBlocking {
        val pending = listOf(KbRow("early_blight", "Gamma 10% EC", "100 ml per hectare", 1, true, reviewed = false))
        val ctx = LayaContext("2026-10-20T09:00:00+05:30", null, true, null, emptyList(), emptyList(), null)
        val r = KbTools(pending, retriever, embedder, { "Early blight" })
            .run(Understanding.Call("t", "get_treatment_info", mapOf("label" to "early_blight")), ctx) as ToolResult.Ok
        assertTrue(answers(null).treatment(r, Lang.EN).text.startsWith(templates.safety("T_NO_KB_CONSULT", Lang.EN)))
    }
}
