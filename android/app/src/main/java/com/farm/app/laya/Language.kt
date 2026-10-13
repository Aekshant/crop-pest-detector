package com.farm.app.laya

/** Reply language, decided from the farmer's own words (never by the SLM: L5 showed it drifts). */
enum class Lang { EN, HI, HINGLISH }

object LanguageDetector {
    // Roman-script Hindi words; the same idea as evals/score_laya.py.
    private val HINGLISH = setOf(
        "hai", "hain", "ho", "tha", "thi", "ka", "ki", "ke", "ko", "kya", "kyu", "kyun", "karo", "karu", "kare", "karein",
        "karna", "kar", "nahi", "nahin", "aap", "aapka", "mein", "me", "se", "par", "pe", "hoga", "hogi", "raha", "rahi",
        "rahe", "yeh", "ye", "woh", "wo", "kaise", "kab", "kitna", "kitne", "kitni", "dawai", "dawa", "khet", "paudha",
        "paudhe", "patti", "patte", "bimari", "ilaj", "ilaaj", "liye", "abhi", "phir", "fir", "baad", "din", "chahiye",
        "dijiye", "sakte", "lagao", "dekho", "batao", "dikhao", "mera", "meri", "tamatar", "jhulsa", "makkhi", "mausam",
        "baarish", "theek", "accha", "kal", "aaj", "parso", "subah", "shaam", "baje", "yaad", "dilana", "likh", "likho",
        "shukriya", "dhanyavad", "namaste", "bhai", "ji", "kheti", "upay", "kaunsi", "daalu", "wala", "wale", "aur",
        "kholo", "dikhata", "dikhati", "hoon", "hun", "hafte", "kaisa", "kaisi", "rahega", "rahegi", "ek", "likhna",
        "dekhna", "dekh", "lo", "hoga", "batana", "chal", "gaya", "gayi", "laga", "kuch",
    )

    fun detect(text: String): Lang {
        val devanagari = text.count { it in 'ऀ'..'ॿ' }
        val latin = text.count { it.isLetter() && it.code < 128 }
        if (devanagari > latin) return Lang.HI
        val words = Regex("[a-z]+").findAll(text.lowercase()).map { it.value }.toList()
        val hits = words.count { it in HINGLISH }
        // Short messages ("shukriya", "hello laya") need only one Hindi word.
        return if (hits >= 2 || (hits >= 1 && words.size <= 3)) Lang.HINGLISH else Lang.EN
    }
}
