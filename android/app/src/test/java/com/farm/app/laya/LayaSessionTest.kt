package com.farm.app.laya

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** L6 done-when: unit tests cover every transition of the Laya state machine. */
@OptIn(ExperimentalCoroutinesApi::class)
class LayaSessionTest {
    private val repo = File("../..").canonicalFile
    private val schema = ToolSchema.parse(File(repo, "ml/bundle/tools_schema_v1.json").readText())
    private val grounding = Grounding.fromLabelsJson(File(repo, "ml/labels_v1.json").readText())

    private val ctx = LayaContext(
        now = "2026-10-20T09:00:00+05:30", location = "Nashik, Maharashtra", online = true, currentField = "f1",
        fields = listOf(Field("f1", "Upar wala khet"), Field("f2", "Nadi wala khet")),
        recentScans = listOf(Scan("s101", "f1", "early_blight", "2026-10-18"), Scan("s103", "f1", "early_blight", "2026-10-11")),
        lastDiagnosis = null,
    )

    /** Scripted SLM: returns the queued answers in order, optionally after a (virtual) delay. */
    private class FakeModel(vararg answers: Pair<Long, Understanding>) : Understander {
        val queue = ArrayDeque(answers.toList())
        val requests = mutableListOf<UnderstandRequest>()
        override suspend fun understand(request: UnderstandRequest): Understanding {
            requests += request
            val (delayMs, answer) = queue.removeFirst()
            delay(delayMs)
            return answer
        }
    }

    private class FakeTools(vararg results: ToolResult) : ToolRunner {
        val queue = ArrayDeque(results.toList())
        val calls = mutableListOf<Understanding.Call>()
        override suspend fun run(call: Understanding.Call, context: LayaContext): ToolResult {
            calls += call
            return queue.removeFirst()
        }
    }

    private fun TestScope.session(model: Understander, tools: ToolRunner = FakeTools()) =
        LayaSession(schema, grounding, model, tools, Clock { testScheduler.currentTime })

    private fun call(tool: String, vararg args: Pair<String, Any?>) = Understanding.Call(tool, tool, mapOf(*args))
    private fun ok(tool: String, followUp: Understanding.Call? = null) = ToolResult.Ok(tool, mapOf("ok" to true), followUp)
    private infix fun Long.then(u: Understanding) = this to u

    // UNDERSTAND -> RESPOND -> DONE
    @Test fun directReply() = runTest {
        val s = session(FakeModel(100L then Understanding.Respond("greeting")))
        val out = s.onUserMessage("namaste laya, kaise ho aap", ctx)  // not an exact fast intent -> model
        assertEquals(TurnOutcome.Answer(Lang.HINGLISH, "greeting", emptyList()), out)
        assertEquals(listOf("UNDERSTAND->RESPOND", "RESPOND->DONE"), s.transitions)
        assertEquals(LayaSession.State.IDLE, s.state)
    }

    // UNDERSTAND -> CALL_TOOL -> VALIDATE -> RESPOND -> DONE
    @Test fun toolCallAnswered() = runTest {
        val tools = FakeTools(ok("get_treatment_info"))
        val s = session(FakeModel(100L then call("get_treatment_info", "label" to "early_blight")), tools)
        val out = s.onUserMessage("agati jhulsa ka ilaaj batao", ctx) as TurnOutcome.Answer
        assertEquals("tool_result", out.kind)
        assertEquals(Lang.HINGLISH, out.lang)
        assertEquals(listOf("UNDERSTAND->CALL_TOOL", "CALL_TOOL->VALIDATE", "VALIDATE->RESPOND", "RESPOND->DONE"), s.transitions)
        assertEquals(1, tools.calls.size)
    }

    // UNDERSTAND -> CLARIFY (model asks) -> UNDERSTAND on the reply -> ... -> DONE
    @Test fun modelClarifiesThenContinues() = runTest {
        val tools = FakeTools(ok("get_treatment_info"))
        val model = FakeModel(100L then Understanding.Clarify("Kaunsi bimari?"),
            100L then call("get_treatment_info", "label" to "late_blight"))
        val s = session(model, tools)
        val ask = s.onUserMessage("dawai batao", ctx) as TurnOutcome.Ask
        assertEquals("Kaunsi bimari?", ask.question)
        assertEquals(LayaSession.State.AWAITING_CLARIFICATION, s.state)
        val out = s.onUserMessage("pachheti jhulsa", ctx)
        assertTrue(out is TurnOutcome.Answer)
        assertEquals(listOf("UNDERSTAND->CLARIFY", "UNDERSTAND->CALL_TOOL", "CALL_TOOL->VALIDATE", "VALIDATE->RESPOND",
            "RESPOND->DONE"), s.transitions)
        // The reply and the question are both in the conversation the model sees.
        assertEquals(listOf("dawai batao", "Kaunsi bimari?", "pachheti jhulsa"), model.requests[1].messages.map { it.text })
    }

    // Missing detail (rule, not the model): UNDERSTAND -> CLARIFY with a slot
    @Test fun missingTimeIsAskedForNotGuessed() = runTest {
        val tools = FakeTools(ok("create_reminder"))
        val guess = call("create_reminder", "when" to "2026-10-21T08:00:00+05:30", "text" to "spray")
        val s = session(FakeModel(100L then guess, 100L then guess), tools)
        val ask = s.onUserMessage("Remind me to spray", ctx) as TurnOutcome.Ask
        assertEquals("when", ask.slot)
        assertTrue(tools.calls.isEmpty())  // nothing ran on a guessed time
        // With the time given, the reminder goes to confirmation.
        val confirm = s.onUserMessage("tomorrow morning", ctx)
        assertTrue(confirm is TurnOutcome.Confirm)
    }

    // CLARIFY limit: a third clarification -> template fallback
    @Test fun clarifyLimit() = runTest {
        val q = Understanding.Clarify("?")
        val s = session(FakeModel(1L then q, 1L then q, 1L then q))
        assertTrue(s.onUserMessage("help", ctx) is TurnOutcome.Ask)
        assertTrue(s.onUserMessage("what", ctx) is TurnOutcome.Ask)
        assertEquals(TurnOutcome.Fallback(Lang.EN, FallbackReason.CLARIFY_LIMIT), s.onUserMessage("hmm", ctx))
        assertEquals("UNDERSTAND->FALLBACK", s.transitions.last())
        assertEquals(LayaSession.State.IDLE, s.state)
    }

    // 8 s generation budget: one slow call
    @Test fun slowModelHitsTimeBudget() = runTest {
        val s = session(FakeModel(9_000L then Understanding.Respond("greeting")))
        assertEquals(TurnOutcome.Fallback(Lang.EN, FallbackReason.TIME_BUDGET), s.onUserMessage("tell me about my farm", ctx))
        assertEquals(8_000L, testScheduler.currentTime)  // cut off at the budget, not after 9 s
    }

    // 8 s budget is per turn: two 5 s calls across a clarification exceed it
    @Test fun budgetAccumulatesAcrossClarification() = runTest {
        val s = session(FakeModel(5_000L then Understanding.Clarify("?"), 5_000L then Understanding.Respond("thanks")))
        assertTrue(s.onUserMessage("help me", ctx) is TurnOutcome.Ask)
        assertEquals(FallbackReason.TIME_BUDGET, (s.onUserMessage("ok", ctx) as TurnOutcome.Fallback).reason)
    }

    @Test fun modelErrorAndBadOutput() = runTest {
        val failing = session(Understander { throw IllegalStateException("model crashed") })
        assertEquals(FallbackReason.MODEL_ERROR, (failing.onUserMessage("tell me about my farm", ctx) as TurnOutcome.Fallback).reason)
        val weird = session(FakeModel(1L then Understanding.Respond("chitchat")))
        assertEquals(FallbackReason.MODEL_ERROR, (weird.onUserMessage("tell me about my farm", ctx) as TurnOutcome.Fallback).reason)
    }

    // CALL_TOOL validates args against the tool schema before running anything
    @Test fun invalidToolCallsNeverRun() = runTest {
        for (bad in listOf(call("buy_pesticide"), call("get_treatment_info", "label" to "powdery_mildew"),
            call("get_scan_history", "days" to 0), call("compare_scans", "scan_a" to "s101"))) {
            val tools = FakeTools()
            val s = session(FakeModel(1L then bad), tools)
            assertEquals(bad.tool, FallbackReason.INVALID_TOOL_CALL, (s.onUserMessage("x", ctx) as TurnOutcome.Fallback).reason)
            assertTrue(tools.calls.isEmpty())
        }
    }

    // Offline: network tools are not offered, and a network call becomes an offline reply
    @Test fun offlineNetworkTool() = runTest {
        val offline = ctx.copy(online = false)
        val model = FakeModel(1L then call("get_weather", "location" to "Nashik"))
        val s = session(model)
        val out = s.onUserMessage("kal baarish hogi kya", offline)
        assertEquals("offline", (out as TurnOutcome.Answer).kind)
        assertFalse("get_weather" in model.requests[0].allowedTools)
        assertTrue("get_weather" in session(FakeModel()).let { _ -> schema.tools.keys })
    }

    // VALIDATE -> CLARIFY: not enough evidence -> ask for a photo
    @Test fun insufficientEvidenceAsksForPhoto() = runTest {
        val s = session(FakeModel(1L then call("compare_scans", "scan_a" to "s103", "scan_b" to "s101")),
            FakeTools(ToolResult.NeedsPhoto("only one scan")))
        val out = s.onUserMessage("Is the blight getting worse?", ctx) as TurnOutcome.Ask
        assertEquals("photo", out.slot)
        assertEquals("VALIDATE->CLARIFY", s.transitions.last())
    }

    // VALIDATE -> FALLBACK: tool failed or threw
    @Test fun toolFailure() = runTest {
        val s = session(FakeModel(1L then call("get_scan_history")), FakeTools(ToolResult.Failed("db error")))
        assertEquals(FallbackReason.TOOL_FAILED, (s.onUserMessage("show my scans", ctx) as TurnOutcome.Fallback).reason)
        val throwing = session(FakeModel(1L then call("get_scan_history"))) { _, _ -> error("boom") }
        assertEquals(FallbackReason.TOOL_FAILED, (throwing.onUserMessage("show my scans", ctx) as TurnOutcome.Fallback).reason)
    }

    // VALIDATE -> CALL_TOOL (follow-up), and the 3-tool limit
    @Test fun followUpChainAndToolLimit() = runTest {
        val next = call("get_treatment_info", "label" to "early_blight")
        val two = session(FakeModel(1L then call("scan_image")), FakeTools(ok("scan_image", next), ok("get_treatment_info")))
        val out = two.onUserMessage("check my leaf for early blight", ctx) as TurnOutcome.Answer
        assertEquals(2, out.results.size)
        assertTrue("VALIDATE->CALL_TOOL" in two.transitions)

        val loop = call("get_scan_history")
        val tools = FakeTools(ok("get_scan_history", loop), ok("get_scan_history", loop), ok("get_scan_history", loop))
        val s = session(FakeModel(1L then loop), tools)
        assertEquals(FallbackReason.TOOL_LIMIT, (s.onUserMessage("show scans", ctx) as TurnOutcome.Fallback).reason)
        assertEquals(3, tools.calls.size)
    }

    // Side effects: CALL_TOOL waits for a confirmation tap
    @Test fun sideEffectNeedsConfirmation() = runTest {
        val note = call("record_observation", "text" to "sprayed neem oil on the river plot")
        val tools = FakeTools(ok("record_observation"))
        val s = session(FakeModel(1L then note), tools)
        val confirm = s.onUserMessage("Note that I sprayed neem oil on the river plot", ctx)
        assertEquals(TurnOutcome.Confirm(Lang.EN, note), confirm)
        assertEquals(LayaSession.State.AWAITING_CONFIRMATION, s.state)
        assertTrue(tools.calls.isEmpty())
        assertEquals("tool_result", (s.onConfirmation(true, ctx) as TurnOutcome.Answer).kind)
        assertEquals(1, tools.calls.size)

        val declined = session(FakeModel(1L then note), FakeTools())
        declined.onUserMessage("Note that I sprayed neem oil on the river plot", ctx)
        assertEquals("cancelled", (declined.onConfirmation(false, ctx) as TurnOutcome.Answer).kind)
    }

    @Test(expected = IllegalStateException::class)
    fun confirmationWithoutPendingCall() = runTest {
        session(FakeModel()).onConfirmation(true, ctx)
    }

    // A new message while a confirmation is pending starts a new turn (counters reset)
    @Test fun newTurnResetsCounters() = runTest {
        val q = Understanding.Clarify("?")
        val note = call("record_observation", "text" to "sprayed neem oil on the river plot")
        val s = session(FakeModel(1L then q, 1L then note, 1L then q, 1L then q))
        s.onUserMessage("note this", ctx)                                      // clarification 1
        assertTrue(s.onUserMessage("sprayed neem oil on the river plot", ctx) is TurnOutcome.Confirm)
        // Ignoring the confirm and asking something new: two fresh clarifications are allowed again.
        assertTrue(s.onUserMessage("help", ctx) is TurnOutcome.Ask)
        assertTrue(s.onUserMessage("again", ctx) is TurnOutcome.Ask)
    }
}
