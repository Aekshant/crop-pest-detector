package com.farm.app.bundle

import java.io.File
import java.security.MessageDigest

sealed interface UpdateResult {
    data class UpToDate(val version: String) : UpdateResult
    data class Incompatible(val version: String, val reason: String) : UpdateResult
    data class BadSignature(val reason: String) : UpdateResult
    data class Failed(val reason: String) : UpdateResult
    data class SmokeFailed(val version: String, val reason: String) : UpdateResult
    data class Updated(val version: String, val downloadedBytes: Long, val reused: Int) : UpdateResult
}

/** Checks a candidate bundle directory before it goes live; null = pass, else the reason. */
fun interface SmokeTest {
    fun check(dir: File, manifest: BundleManifest): String?
}

/**
 * The design doc's bundle update flow (LLD "Model bundle lifecycle"):
 *  1. fetch the manifest (the worker only runs on unmetered networks or while charging)
 *  2. verify its Ed25519 signature with a key compiled into the app; else discard
 *  3. check min_app_version, tool_schema_version and the device tier; else keep the current bundle
 *  4. get only changed artifacts (by sha256: unchanged ones are copied from the current
 *     bundle), resumable, each verified by size and sha256
 *  5. smoke test the candidate
 *  6. pass: switch atomically; fail: delete the candidate and report
 *  7. the old bundle stays as "previous"
 * Blocking; run it off the main thread (BundleWorker). Not reentrant.
 */
class BundleUpdater(
    private val source: BundleSource,
    private val store: BundleStore,
    private val trustedKeys: Map<String, ByteArray>,
    private val appVersionCode: Int,
    private val supportedToolSchemas: Set<Int>,
    private val tier: String,
    private val smokeTest: SmokeTest,
) {
    fun update(): UpdateResult {
        val envelope = runCatching { source.manifest() }.getOrElse { return UpdateResult.Failed("manifest: ${it.message}") }
        val m = try {
            BundleManifest.verify(envelope, trustedKeys)
        } catch (e: BundleManifest.BadSignature) {
            return UpdateResult.BadSignature(e.message.orEmpty())
        } catch (e: Exception) {
            return UpdateResult.BadSignature("unreadable manifest: ${e.message}")
        }
        if (m.version == store.current()) return UpdateResult.UpToDate(m.version)
        when {
            m.minAppVersion > appVersionCode -> return UpdateResult.Incompatible(m.version, "needs app version ${m.minAppVersion}, have $appVersionCode")
            m.toolSchemaVersion !in supportedToolSchemas -> return UpdateResult.Incompatible(m.version, "tool schema ${m.toolSchemaVersion} not supported")
            tier !in m.tiers -> return UpdateResult.Incompatible(m.version, "not for tier $tier")
        }

        store.cleanup(keepStaging = m.version)  // keep this candidate's partial downloads, drop older ones
        val staging = store.staging(m.version).apply { mkdirs() }
        val current = store.currentDir()
        var downloaded = 0L
        var reused = 0
        for (a in m.artifactsFor(tier)) {
            val dest = File(staging, a.file)
            if (dest.exists() && dest.length() == a.bytes && sha256(dest) == a.sha256) continue
            val same = current?.let { File(it, a.file) }?.takeIf { it.exists() && it.length() == a.bytes && sha256(it) == a.sha256 }
            if (same != null) {
                same.copyTo(dest, overwrite = true)
                reused++
                continue
            }
            val part = File(staging, a.file + ".part")
            val before = if (part.exists()) part.length() else 0L
            try {
                source.download(m.version, a.file, part, a.bytes)
            } catch (e: Exception) {
                return UpdateResult.Failed("${a.file}: ${e.message}")  // .part stays: the next run resumes
            }
            downloaded += part.length() - before
            if (part.length() != a.bytes || sha256(part) != a.sha256) {
                part.delete()
                return UpdateResult.Failed("${a.file}: size or sha256 mismatch")
            }
            check(part.renameTo(dest)) { "rename ${a.file}" }
        }
        File(staging, MANIFEST).writeBytes(envelope)

        val smoke = runCatching { smokeTest.check(staging, m) }.getOrElse { "smoke test crashed: ${it.message}" }
        if (smoke != null) {
            staging.deleteRecursively()
            return UpdateResult.SmokeFailed(m.version, smoke)
        }
        store.activate(m.version)
        return UpdateResult.Updated(m.version, downloaded, reused)
    }

    companion object {
        /** The signed envelope, kept next to the artifacts of each bundle. */
        const val MANIFEST = "manifest.json.env"

        fun sha256(f: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            f.inputStream().use { input ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
