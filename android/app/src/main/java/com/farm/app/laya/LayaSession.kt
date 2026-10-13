package com.farm.app.laya

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout

data class Field(val id: String, val name: String)
data class Scan(val id: String, val fieldId: String, val label: String, val takenAt: String)

/** What the app knows when the farmer speaks (design doc: state lives in the app). */
data class LayaContext(
    val now: String,
    val location: String?,
    val online: Boolean,
    val currentField: String?,
    val fields: List<Field>,
    val recentScans: List<Scan>,
    val lastDiagnosis: String?,
)

data class Message(val fromUser: Boolean, val text: String)

/** The UNDERSTAND step's output (schema-constrained JSON from the SLM, parsed). */
sealed interface Understanding {
    data class Call(val intent: String, val tool: String, val args: Map<String, Any?>) : Understanding
    data class Clarify(val question: String) : Understanding
    data class Respond(val kind: String) : Understanding
}

data class UnderstandRequest(
    val messages: List<Message>,
    val context: LayaContext,
    val toolResults: List<ToolResult.Ok>,
    /** Tools the model may use now (network tools are removed offline). */
    val allowedTools: Set<String>,
)

/** The SLM (edge or cloud). Implementations must be cancellable: the session enforces the time budget. */
fun interface Understander {
    suspend fun understand(request: UnderstandRequest): Understanding
}

sealed interface ToolResult {
    /** [followUp]: a tool the result asks for next (e.g. a scan that found a disease -> treatment info). */
    data class Ok(val tool: String, val data: Map<String, Any?>, val followUp: Understanding.Call? = null) : ToolResult
    /** VALIDATE: not enough evidence to answer (e.g. no recent scan) -> ask for a photo. */
    data class NeedsPhoto(val reason: String) : ToolResult
    data class Failed(val reason: String) : ToolResult
}

fun interface ToolRunner {
    suspend fun run(call: Understanding.Call, context: LayaContext): ToolResult
}

fun interface Clock {
    fun nowMs(): Long
}

enum class FallbackReason { TOOL_LIMIT, CLARIFY_LIMIT, TIME_BUDGET, MODEL_ERROR, INVALID_TOOL_CALL, TOOL_FAILED }

/** What a turn ends with. User-facing sentences come from templates (L8) in [lang]. */
sealed interface TurnOutcome {
    val lang: Lang

    /** RESPOND -> DONE. kind: a respond kind, "tool_result", "offline" or "cancelled". */
    data class Answer(override val lang: Lang, val kind: String, val results: List<ToolResult.Ok>) : TurnOutcome

    /** CLARIFY: one question. [question] when the model asked; [slot] when a rule found a missing detail. */
    data class Ask(override val lang: Lang, val question: String?, val slot: String?) : TurnOutcome

    /** A side-effect tool waits for the farmer's confirmation tap. */
    data class Confirm(override val lang: Lang, val call: Understanding.Call) : TurnOutcome

    /** A hard limit or a failure: answer from a template, never an error screen. */
    data class Fallback(override val lang: Lang, val reason: FallbackReason) : TurnOutcome
}

data class Limits(val maxToolCalls: Int = 3, val maxClarifications: Int = 2, val generationBudgetMs: Long = 8_000)

/**
 * Laya's bounded state machine for one conversation (design doc, LLD Laya):
 *
 *   UNDERSTAND -> CALL_TOOL | CLARIFY | RESPOND
 *   CALL_TOOL  -> VALIDATE                (args validated against the tool schema first)
 *   VALIDATE   -> RESPOND | CLARIFY | CALL_TOOL (follow-up)
 *   RESPOND    -> DONE
 *   CLARIFY    -> UNDERSTAND              (on the farmer's reply)
 *
 * Hard limits per turn: 3 tool calls, 2 clarifications, 8 s of total generation.
 * Any limit or failure -> template fallback. Not thread-safe: one session per
 * conversation, used from one coroutine at a time.
 */
class LayaSession(
    private val schema: ToolSchema,
    private val grounding: Grounding,
    private val understander: Understander,
    private val tools: ToolRunner,
    private val clock: Clock,
    private val limits: Limits = Limits(),
) {
    enum class State { IDLE, AWAITING_CLARIFICATION, AWAITING_CONFIRMATION }

    var state = State.IDLE
        private set

    /** Every transition taken, e.g. "UNDERSTAND->CALL_TOOL" (for tests and the debug screen). */
    val transitions = mutableListOf<String>()

    private val messages = mutableListOf<Message>()
    private val results = mutableListOf<ToolResult.Ok>()
    private var toolCalls = 0
    private var clarifications = 0
    private var generationMs = 0L
    private var lang = Lang.EN
    private var pending: Understanding.Call? = null

    suspend fun onUserMessage(text: String, context: LayaContext): TurnOutcome {
        if (state != State.AWAITING_CLARIFICATION) startTurn(text)  // a reply to a confirmation prompt starts over
        else messages += Message(true, text)
        state = State.IDLE
        return understand(context)
    }

    /** The farmer tapped confirm (or cancel) on a side-effect tool. */
    suspend fun onConfirmation(confirmed: Boolean, context: LayaContext): TurnOutcome {
        val call = checkNotNull(pending) { "no confirmation pending" }
        check(state == State.AWAITING_CONFIRMATION)
        pending = null
        state = State.IDLE
        if (!confirmed) return done(TurnOutcome.Answer(lang, "cancelled", results.toList()))
        return run(call, context)
    }

    private fun startTurn(text: String) {
        messages.clear()
        results.clear()
        toolCalls = 0
        clarifications = 0
        generationMs = 0
        pending = null
        lang = LanguageDetector.detect(text)
        messages += Message(true, text)
    }

    private fun go(from: String, to: String) {
        transitions += "$from->$to"
    }

    private suspend fun understand(context: LayaContext): TurnOutcome {
        val remaining = limits.generationBudgetMs - generationMs
        if (remaining <= 0) return fallback("UNDERSTAND", FallbackReason.TIME_BUDGET)
        val allowed = schema.tools.values.filter { context.online || !it.needsNetwork }.map { it.name }.toSet()
        val start = clock.nowMs()
        val u = try {
            withTimeout(remaining) {
                understander.understand(UnderstandRequest(messages.toList(), context, results.toList(), allowed))
            }
        } catch (e: TimeoutCancellationException) {
            generationMs += clock.nowMs() - start
            return fallback("UNDERSTAND", FallbackReason.TIME_BUDGET)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            return fallback("UNDERSTAND", FallbackReason.MODEL_ERROR)
        }
        generationMs += clock.nowMs() - start
        return when (u) {
            is Understanding.Respond ->
                if (u.kind in schema.respondKinds) { go("UNDERSTAND", "RESPOND"); done(TurnOutcome.Answer(lang, u.kind, results.toList())) }
                else fallback("UNDERSTAND", FallbackReason.MODEL_ERROR)
            is Understanding.Clarify -> clarify("UNDERSTAND", u.question, null)
            is Understanding.Call -> callTool("UNDERSTAND", u, context)
        }
    }

    private suspend fun callTool(from: String, call: Understanding.Call, context: LayaContext): TurnOutcome {
        val spec = schema.tools[call.tool] ?: return fallback(from, FallbackReason.INVALID_TOOL_CALL)
        if (spec.needsNetwork && !context.online) {
            go(from, "RESPOND")
            return done(TurnOutcome.Answer(lang, "offline", results.toList()))
        }
        if (schema.validate(call) != null) return fallback(from, FallbackReason.INVALID_TOOL_CALL)
        val userText = messages.filter { it.fromUser }.joinToString(" ") { it.text }
        grounding.missing(call, userText, context)?.let { slot -> return clarify(from, null, slot) }
        if (toolCalls >= limits.maxToolCalls) return fallback(from, FallbackReason.TOOL_LIMIT)
        go(from, "CALL_TOOL")
        if (spec.sideEffect) {
            pending = call
            state = State.AWAITING_CONFIRMATION
            return TurnOutcome.Confirm(lang, call)
        }
        return run(call, context)
    }

    private suspend fun run(call: Understanding.Call, context: LayaContext): TurnOutcome {
        toolCalls++
        val result = try {
            tools.run(call, context)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            ToolResult.Failed(e.message ?: "tool error")
        }
        go("CALL_TOOL", "VALIDATE")
        return when (result) {
            is ToolResult.Ok -> {
                results += result
                val next = result.followUp
                if (next != null) callTool("VALIDATE", next, context)
                else { go("VALIDATE", "RESPOND"); done(TurnOutcome.Answer(lang, "tool_result", results.toList())) }
            }
            is ToolResult.NeedsPhoto -> clarify("VALIDATE", null, "photo")
            is ToolResult.Failed -> fallback("VALIDATE", FallbackReason.TOOL_FAILED)
        }
    }

    private fun clarify(from: String, question: String?, slot: String?): TurnOutcome {
        if (clarifications >= limits.maxClarifications) return fallback(from, FallbackReason.CLARIFY_LIMIT)
        clarifications++
        go(from, "CLARIFY")
        state = State.AWAITING_CLARIFICATION
        if (question != null) messages += Message(false, question)
        return TurnOutcome.Ask(lang, question, slot)
    }

    private fun done(outcome: TurnOutcome): TurnOutcome {
        go("RESPOND", "DONE")
        state = State.IDLE
        return outcome
    }

    private fun fallback(from: String, reason: FallbackReason): TurnOutcome {
        go(from, "FALLBACK")
        state = State.IDLE
        pending = null
        return TurnOutcome.Fallback(lang, reason)
    }
}
