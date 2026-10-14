package com.farm.app.bundle

import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Bundles on disk:
 *   <root>/versions/<version>/   activated bundles (current and previous only)
 *   <root>/staging/<version>/    a candidate being downloaded / smoke-tested (partial files resume)
 *   <root>/bundle_state.json     {"current": v, "previous": v}: switched by an atomic rename
 * (The design doc's "one row in a bundle_state table"; a file renamed atomically is the same guarantee.)
 */
class BundleStore(val root: File) {
    private val state = File(root, "bundle_state.json")

    fun current(): String? = read()?.optString("current")?.takeIf { it.isNotEmpty() }
    fun previous(): String? = read()?.optString("previous")?.takeIf { it.isNotEmpty() }
    fun dir(version: String) = File(root, "versions/$version")
    fun staging(version: String) = File(root, "staging/$version")
    fun currentDir(): File? = current()?.let(::dir)?.takeIf { it.isDirectory }

    /** Moves the verified candidate into versions/, makes it current, keeps the old current as previous. */
    fun activate(version: String) {
        val target = dir(version)
        if (target.exists()) target.deleteRecursively()
        target.parentFile!!.mkdirs()
        Files.move(staging(version).toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
        val old = current()
        val tmp = File(root, "bundle_state.json.tmp")
        tmp.writeText(JSONObject().put("current", version).put("previous", old?.takeIf { it != version } ?: previous().orEmpty()).toString())
        Files.move(tmp.toPath(), state.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        cleanup()
    }

    /** Deletes every version that is neither current nor previous, and stale candidates. */
    fun cleanup(keepStaging: String? = null) {
        val keep = setOfNotNull(current(), previous())
        File(root, "versions").listFiles()?.filter { it.name !in keep }?.forEach { it.deleteRecursively() }
        File(root, "staging").listFiles()?.filter { it.name != keepStaging }?.forEach { it.deleteRecursively() }
    }

    private fun read(): JSONObject? = state.takeIf { it.exists() }?.let { runCatching { JSONObject(it.readText()) }.getOrNull() }
}
