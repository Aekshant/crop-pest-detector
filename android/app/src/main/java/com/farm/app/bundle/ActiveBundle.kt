package com.farm.app.bundle

import android.content.Context
import com.farm.app.BuildConfig
import java.io.File

/**
 * An activated model bundle as the app uses it. [candidates] lists the bundles to try,
 * current first and then previous, each re-checked when loaded: the signature of its
 * stored manifest and the size of every artifact. Hashes were verified at download;
 * re-hashing ~130 MB at every start would cost seconds. A bundle that fails these
 * checks, or later fails to load, is skipped: the previous bundle or the APK's
 * built-in (factory) models take over.
 */
class ActiveBundle(val version: String, val dir: File, val manifest: BundleManifest, val role: String) {
    /** The artifact's file, or null if this bundle doesn't ship it (e.g. no GPU model for tier C). */
    fun file(artifact: String): File? = manifest.artifacts.firstOrNull { it.name == artifact }
        ?.let { File(dir, it.file) }?.takeIf { it.isFile }

    val hasVision get() = file("vision") != null && file("vision_config") != null && file("labels") != null
    val hasKb get() = file("kb") != null && file("kb_embedder") != null && file("kb_vocab") != null

    override fun toString() = "$version ($role)"

    companion object {
        fun candidates(store: BundleStore, trustedKeys: Map<String, ByteArray>, tier: String): List<ActiveBundle> =
            listOfNotNull(store.current()?.let { it to "current" }, store.previous()?.let { it to "previous" })
                .mapNotNull { (version, role) -> load(store.dir(version), version, role, trustedKeys, tier) }

        /** Activated bundles that fail the load-time checks (damaged on disk or no longer trusted), e.g. "2026.10.15 (current)". */
        fun rejected(store: BundleStore, trustedKeys: Map<String, ByteArray>, tier: String): List<String> {
            val ok = candidates(store, trustedKeys, tier).map { it.version }.toSet()
            return listOfNotNull(store.current()?.let { it to "current" }, store.previous()?.let { it to "previous" })
                .filter { it.first !in ok }.map { "${it.first} (${it.second})" }
        }

        fun rejected(context: Context, tier: String = DeviceTier.current(context)): List<String> =
            rejected(BundleWorker.store(context), BundleManifest.keys(BuildConfig.BUNDLE_PUBLIC_KEYS), tier)

        fun candidates(context: Context, tier: String = DeviceTier.current(context)): List<ActiveBundle> =
            candidates(BundleWorker.store(context), BundleManifest.keys(BuildConfig.BUNDLE_PUBLIC_KEYS), tier)

        private fun load(dir: File, version: String, role: String, keys: Map<String, ByteArray>, tier: String): ActiveBundle? {
            val envelope = File(dir, BundleUpdater.MANIFEST).takeIf { it.isFile } ?: return null
            val m = runCatching { BundleManifest.verify(envelope.readBytes(), keys) }.getOrNull() ?: return null
            if (m.version != version) return null
            val intact = m.artifactsFor(tier).all { File(dir, it.file).let { f -> f.isFile && f.length() == it.bytes } }
            return if (intact) ActiveBundle(version, dir, m, role) else null
        }
    }
}
