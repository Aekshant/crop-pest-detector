package com.farm.app.laya

/**
 * L7: fast intents (design doc, Laya request routing). Short, exact commands in
 * Hindi, Hinglish or English ("scan karo", "history dikhao", "namaste") are
 * answered without the SLM. The whole message must be one of the commands
 * (after dropping filler like "please" or "ji"), so free-form questions are
 * never hijacked: they go to the state machine.
 */
object FastIntentMatcher {
    private val FILLER = setOf("please", "plz", "pls", "laya", "ji", "zara", "जरा", "ज़रा", "लाया", "जी", "कृपया")

    private val SCAN = setOf(
        "scan", "scan karo", "scan kar do", "scan kijiye", "scan kare", "photo lo", "photo le lo", "photo khicho",
        "jaanch karo", "janch karo", "jaanch kar do", "patti jaancho", "patti check karo", "check karo", "camera kholo",
        "check leaf", "check my leaf", "scan leaf", "scan a leaf", "scan my plant", "take photo", "take a photo",
        "open camera", "open the camera",
        "स्कैन", "स्कैन करो", "स्कैन कीजिए", "जाँच करो", "जांच करो", "जाँच कीजिए", "फोटो लो", "फोटो खींचो", "कैमरा खोलो",
        "पत्ती जाँचो", "पत्ती जांचो",
    )
    private val HISTORY = setOf(
        "history", "history dikhao", "history batao", "purane scan dikhao", "purani jaanch dikhao", "scan history",
        "my scans", "show history", "show my history", "show my scans", "past scans",
        "इतिहास", "इतिहास दिखाओ", "पुरानी जाँच दिखाओ", "पुरानी जांच दिखाओ", "पुराने स्कैन दिखाओ",
    )
    private val WEATHER = setOf(
        "mausam", "mausam batao", "mausam kaisa hai", "aaj ka mausam", "weather", "weather today", "today's weather",
        "मौसम", "मौसम बताओ", "मौसम कैसा है", "आज का मौसम",
    )
    private val GREETING = setOf(
        "hi", "hello", "hey", "namaste", "namaskar", "ram ram", "good morning", "नमस्ते", "नमस्कार", "राम राम",
    )
    private val THANKS = setOf(
        "thanks", "thank you", "thank you so much", "shukriya", "dhanyavad", "dhanyawad", "bahut shukriya",
        "धन्यवाद", "शुक्रिया", "बहुत धन्यवाद",
    )

    fun normalize(text: String): String {
        val words = Grounding.normalize(text).replace(Regex("""[^\p{L}\p{M}\p{N}' ]"""), " ")
            .split(" ").filter { it.isNotBlank() && it !in FILLER }
        return words.joinToString(" ")
    }

    /** The fast answer for [text], or null when it isn't a known short command. */
    fun match(text: String, context: LayaContext): Understanding? {
        val t = normalize(text)
        return when (t) {
            in SCAN -> Understanding.Call("scan", "scan_image", emptyMap())
            in HISTORY -> Understanding.Call("history", "get_scan_history", emptyMap())
            in WEATHER -> context.location?.let { Understanding.Call("weather", "get_weather", mapOf("location" to it)) }
            in GREETING -> Understanding.Respond("greeting")
            in THANKS -> Understanding.Respond("thanks")
            else -> null
        }
    }
}
