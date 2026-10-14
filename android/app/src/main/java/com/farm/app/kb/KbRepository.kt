package com.farm.app.kb

import android.content.Context
import android.util.Log
import com.farm.app.BuildConfig
import com.farm.app.bundle.ActiveBundle

/**
 * The knowledge base from the active model bundle (current, else previous). The APK has
 * no built-in KB: until a bundle with one is activated, [obtain] returns null and Laya
 * answers treatment questions with the "ask your KVK" template. Load off the main thread.
 */
object KbRepository {
    private const val TAG = "KbRepository"
    @Volatile private var loaded: Pair<String, KbBundle.Loaded>? = null

    @Synchronized
    fun obtain(context: Context): KbBundle.Loaded? {
        val bundles = runCatching { ActiveBundle.candidates(context) }.getOrDefault(emptyList()).filter { it.hasKb }
        loaded?.let { (version, kb) -> if (bundles.firstOrNull()?.version == version) return kb else kb.close() }
        loaded = null
        val labels = context.assets.open("vision/labels_v1.json").bufferedReader().use { it.readText() }
        for (b in bundles) {
            try {
                // Debug builds may use rows still pending agronomist review; release builds only approved rows.
                val kb = KbBundle.open(b.file("kb")!!, b.file("kb_embedder")!!, b.file("kb_vocab")!!, labels, includeUnreviewed = BuildConfig.DEBUG)
                loaded = b.version to kb
                Log.i(TAG, "KB ${kb.pack.meta["kb_version"]} from bundle $b")
                return kb
            } catch (e: Exception) {
                Log.e(TAG, "KB from bundle $b failed", e)
            }
        }
        return null
    }

    /** "kb 2026-10-14 from 2026.10.15 (current)", or null if none is loaded. */
    fun describe(): String? = loaded?.let { (v, kb) -> "${kb.pack.meta["kb_version"]} from bundle $v" }
}
