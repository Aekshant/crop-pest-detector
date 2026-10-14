package com.farm.app.bundle

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** Where bundles come from: <base>/manifest.json and <base>/<version>/<file>. */
interface BundleSource {
    fun manifest(): ByteArray

    /** Downloads [file] of [version] into [dest], continuing from dest's current length (resumable). */
    fun download(version: String, file: String, dest: File, expectedBytes: Long)
}

/** Static HTTPS hosting (CDN / bucket). Resumes with HTTP Range; restarts if the server ignores it. */
class HttpBundleSource(private val baseUrl: String, private val timeoutMs: Int = 30_000) : BundleSource {
    override fun manifest(): ByteArray = open("$baseUrl/manifest.json").let { c ->
        try {
            check(c.responseCode == 200) { "manifest: HTTP ${c.responseCode}" }
            c.inputStream.use { it.readBytes() }
        } finally {
            c.disconnect()
        }
    }

    override fun download(version: String, file: String, dest: File, expectedBytes: Long) {
        val have = if (dest.exists()) dest.length() else 0L
        if (have == expectedBytes && expectedBytes > 0) return
        if (have > expectedBytes) dest.delete()
        val start = if (dest.exists()) dest.length() else 0L
        val c = open("$baseUrl/$version/$file").apply { if (start > 0) setRequestProperty("Range", "bytes=$start-") }
        try {
            val append = when (c.responseCode) {
                206 -> true
                200 -> false  // server ignored Range: start over
                else -> throw IOException("$file: HTTP ${c.responseCode}")
            }
            c.inputStream.use { input ->
                FileOutputStream(dest, append).use { out ->
                    val buf = ByteArray(1 shl 16)
                    var total = if (append) start else 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        total += n
                        if (total > expectedBytes) throw IOException("$file: larger than the manifest says")
                        out.write(buf, 0, n)
                    }
                }
            }
        } finally {
            c.disconnect()
        }
    }

    private fun open(url: String): HttpURLConnection {
        require(url.startsWith("https://") || url.startsWith("http://127.0.0.1") || url.startsWith("http://localhost")) {
            "bundles only over HTTPS"
        }
        return (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = timeoutMs
            readTimeout = timeoutMs
            instanceFollowRedirects = true
        }
    }
}
