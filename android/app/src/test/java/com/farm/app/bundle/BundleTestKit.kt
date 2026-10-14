package com.farm.app.bundle

import com.sun.net.httpserver.HttpServer
import org.bouncycastle.crypto.params.Ed25519PrivateKeyParameters
import org.bouncycastle.crypto.signers.Ed25519Signer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.InetSocketAddress
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import java.util.concurrent.atomic.AtomicLong

/** Signs bundles and serves them over real HTTP (with Range, and a switch to cut downloads short). */
class BundleTestKit(val root: File) {
    private val key = Ed25519PrivateKeyParameters(SecureRandom())
    val keyId = "test"
    val trusted = mapOf(keyId to key.generatePublicKey().encoded)

    /** Bytes sent for artifact downloads (not the manifest). */
    val bytesServed = AtomicLong()
    /** If > 0, the next artifact response stops after this many bytes (simulated dropped connection). */
    @Volatile var cutAfter = 0L
    @Volatile var ignoreRange = false

    private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
        createContext("/") { ex ->
            val f = File(root, ex.requestURI.path.removePrefix("/"))
            if (!f.isFile) { ex.sendResponseHeaders(404, -1); ex.close(); return@createContext }
            val bytes = f.readBytes()
            val range = ex.requestHeaders.getFirst("Range")?.takeIf { !ignoreRange }
            val start = range?.removePrefix("bytes=")?.substringBefore('-')?.toLong() ?: 0L
            val body = bytes.copyOfRange(start.toInt(), bytes.size)
            val manifest = f.name == "manifest.json"
            val limit = if (!manifest && cutAfter > 0) minOf(cutAfter, body.size.toLong()).also { cutAfter = 0 } else body.size.toLong()
            ex.sendResponseHeaders(if (range != null) 206 else 200, body.size.toLong())
            try {
                ex.responseBody.write(body, 0, limit.toInt())
                if (!manifest) bytesServed.addAndGet(limit)
            } finally {
                ex.close()  // closing early = the client sees a truncated body
            }
        }
        start()
    }
    val baseUrl = "http://127.0.0.1:${server.address.port}"

    fun close() = server.stop(0)

    /** Writes <root>/<version>/<files> and a signed <root>/manifest.json. */
    fun publish(version: String, files: Map<String, ByteArray>, tiers: Map<String, List<String>> = emptyMap(),
                minAppVersion: Int = 1, toolSchema: Int = 1, corrupt: String? = null) {
        val dir = File(root, version).apply { mkdirs() }
        val arts = JSONArray()
        for ((name, data) in files) {
            File(dir, name).writeBytes(if (name == corrupt) data.reversedArray() else data)
            arts.put(JSONObject().put("name", name.substringBefore('.')).put("file", name).put("sha256", sha(data))
                .put("bytes", data.size).apply { tiers[name]?.let { put("tiers", JSONArray(it)) } })
        }
        val manifest = JSONObject().put("bundle_version", version).put("min_app_version", minAppVersion)
            .put("tool_schema_version", toolSchema).put("tiers", JSONArray(listOf("A", "B", "C"))).put("artifacts", arts)
        File(root, "manifest.json").writeText(envelope(manifest.toString(1).toByteArray()))
    }

    fun envelope(bytes: ByteArray, signer: Ed25519PrivateKeyParameters = key, id: String = keyId): String {
        val sig = Ed25519Signer().run { init(true, signer); update(bytes, 0, bytes.size); generateSignature() }
        return JSONObject().put("key_id", id).put("manifest", Base64.getEncoder().encodeToString(bytes))
            .put("signature", "ed25519:" + Base64.getEncoder().encodeToString(sig)).toString()
    }

    companion object {
        fun sha(b: ByteArray) = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }
        fun data(n: Int, seed: Int) = ByteArray(n) { ((it * 31 + seed * 7) % 251).toByte() }
    }
}
