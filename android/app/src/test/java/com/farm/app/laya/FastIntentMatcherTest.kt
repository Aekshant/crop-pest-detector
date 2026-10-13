package com.farm.app.laya

import kotlinx.coroutines.test.runTest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** L7 done-when: common commands are answered in < 100 ms, without the SLM. */
class FastIntentMatcherTest {
    private val repo = File("../..").canonicalFile
    private val ctx = LayaContext("2026-10-20T09:00:00+05:30", "Nashik, Maharashtra", true, "f1",
        listOf(Field("f1", "Upar wala khet")), emptyList(), null)

    private val common = mapOf(
        "scan karo" to "scan_image", "Scan karo please" to "scan_image", "स्कैन करो" to "scan_image",
        "जाँच करो" to "scan_image", "photo lo" to "scan_image", "Take a photo" to "scan_image", "camera kholo ji" to "scan_image",
        "history dikhao" to "get_scan_history", "इतिहास दिखाओ" to "get_scan_history", "Show my scans" to "get_scan_history",
        "mausam batao" to "get_weather", "मौसम बताओ" to "get_weather", "Weather today?" to "get_weather",
        "namaste laya" to "greeting", "नमस्ते" to "greeting", "Hello!" to "greeting",
        "shukriya" to "thanks", "धन्यवाद" to "thanks", "Thank you so much" to "thanks",
    )

    private fun answer(u: Understanding?) = when (u) {
        is Understanding.Call -> u.tool
        is Understanding.Respond -> u.kind
        else -> null
    }

    @Test fun commonCommandsInThreeLanguages() {
        for ((text, expected) in common) assertEquals(text, expected, answer(FastIntentMatcher.match(text, ctx)))
    }

    @Test fun answersInUnder100ms() {
        repeat(1000) { FastIntentMatcher.match("scan karo", ctx) }  // warm up the JIT
        val t = System.nanoTime()
        val n = 10_000
        repeat(n) { i -> FastIntentMatcher.match(common.keys.elementAt(i % common.size), ctx) }
        val perCallMs = (System.nanoTime() - t) / 1e6 / n
        assertTrue("per call $perCallMs ms", perCallMs < 100.0)
        assertTrue("per call $perCallMs ms should be far below budget", perCallMs < 1.0)
    }

    @Test fun freeFormQuestionsAreNotHijacked() {
        for (text in listOf("scan karo aur batao kya bimari hai", "history of the upper field for the past month",
            "Hi, my leaves have brown spots", "kal baarish hogi kya", "photo lo nadi wale khet ki")) {
            assertNull(text, FastIntentMatcher.match(text, ctx))
        }
    }

    @Test fun weatherWithoutLocationIsNotFast() {
        assertNull(FastIntentMatcher.match("mausam batao", ctx.copy(location = null)))
    }

    /** Whenever a fast intent fires on a real eval case, it must agree with the expected answer. */
    @Test fun neverContradictsTheEval() {
        val cases = File(repo, "evals/laya/cases.jsonl").readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
        var matched = 0
        val wrong = cases.mapNotNull { c ->
            val override = c.optJSONObject("context")
            val loc = if (override != null && override.has("location") && override.isNull("location")) null else "Nashik"
            val u = FastIntentMatcher.match(c.getString("input"), ctx.copy(location = loc)) ?: return@mapNotNull null
            matched++
            val e = c.getJSONObject("expect")
            val expected = when (e.getString("action")) { "tool" -> e.getString("tool"); "respond" -> e.getString("kind"); else -> "clarify" }
            if (answer(u) != expected || c.optJSONObject("context")?.optString("network") == "OFFLINE") "${c.getString("id")}: ${answer(u)} vs $expected" else null
        }
        assertTrue(wrong.joinToString("\n"), wrong.isEmpty())
        assertTrue("fast intents should cover a few eval cases, got $matched", matched >= 3)
    }

    @Test fun sessionSkipsTheModelForFastIntents() = runTest {
        val schema = ToolSchema.parse(File(repo, "ml/bundle/tools_schema_v1.json").readText())
        val grounding = Grounding.fromLabelsJson(File(repo, "ml/labels_v1.json").readText())
        var modelCalls = 0
        val s = LayaSession(schema, grounding, { modelCalls++; Understanding.Respond("greeting") },
            { call, _ -> ToolResult.Ok(call.tool, emptyMap()) }, Clock { 0 })
        assertEquals("tool_result", (s.onUserMessage("scan karo", ctx) as TurnOutcome.Answer).kind)
        assertEquals(listOf("FAST->CALL_TOOL", "CALL_TOOL->VALIDATE", "VALIDATE->RESPOND", "RESPOND->DONE"), s.transitions)
        assertEquals(0, modelCalls)
    }
}
