package com.farm.app.bundle

import org.bouncycastle.crypto.params.Ed25519PublicKeyParameters
import org.bouncycastle.crypto.signers.Ed25519Signer
import org.json.JSONObject
import java.util.Base64

data class Artifact(val name: String, val file: String, val sha256: String, val bytes: Long, val tiers: List<String>?) {
    fun forTier(tier: String) = tiers == null || tier in tiers
}

/**
 * A verified bundle manifest (ml/bundle/build_bundle.py). The envelope is
 * {"key_id", "manifest": base64(manifest JSON bytes), "signature": "ed25519:base64"}:
 * the signature covers the exact bytes, so no JSON canonicalization is involved.
 */
data class BundleManifest(
    val version: String,
    val minAppVersion: Int,
    val toolSchemaVersion: Int,
    val tiers: List<String>,
    val artifacts: List<Artifact>,
) {
    fun artifactsFor(tier: String) = artifacts.filter { it.forTier(tier) }

    class BadSignature(message: String) : Exception(message)

    companion object {
        private val SAFE_NAME = Regex("^[A-Za-z0-9._-]{1,128}$")
        private val SHA256 = Regex("^[0-9a-f]{64}$")
        private val VERSION = Regex("^[0-9A-Za-z._-]{1,64}$")

        /** Verifies [envelope] against [trustedKeys] (key id -> raw Ed25519 public key) and parses it. */
        fun verify(envelope: ByteArray, trustedKeys: Map<String, ByteArray>): BundleManifest {
            val env = runCatching { JSONObject(String(envelope, Charsets.UTF_8)) }.getOrElse { throw BadSignature("not JSON") }
            val keyId = env.optString("key_id")
            val key = trustedKeys[keyId] ?: throw BadSignature("untrusted key id '$keyId'")
            val sig = env.optString("signature")
            if (!sig.startsWith("ed25519:")) throw BadSignature("not an ed25519 signature")
            val bytes = runCatching { Base64.getDecoder().decode(env.getString("manifest")) }.getOrElse { throw BadSignature("bad manifest encoding") }
            val signature = runCatching { Base64.getDecoder().decode(sig.removePrefix("ed25519:")) }.getOrElse { throw BadSignature("bad signature encoding") }
            val ok = Ed25519Signer().run {
                init(false, Ed25519PublicKeyParameters(key, 0))
                update(bytes, 0, bytes.size)
                verifySignature(signature)
            }
            if (!ok) throw BadSignature("signature does not match")
            return parse(bytes)
        }

        /** Parses verified manifest bytes; also refuses unsafe names (defence in depth: they become file paths). */
        internal fun parse(bytes: ByteArray): BundleManifest {
            val m = JSONObject(String(bytes, Charsets.UTF_8))
            val version = m.getString("bundle_version")
            require(VERSION.matches(version)) { "bad bundle_version" }
            val arts = m.getJSONArray("artifacts").let { a -> List(a.length()) { a.getJSONObject(it) } }.map { o ->
                val tiers = o.optJSONArray("tiers")?.let { t -> List(t.length()) { t.getString(it) } }
                Artifact(o.getString("name"), o.getString("file"), o.getString("sha256"), o.getLong("bytes"), tiers).also {
                    require(SAFE_NAME.matches(it.file) && it.file != "manifest.json.env") { "unsafe file name ${it.file}" }
                    require(SHA256.matches(it.sha256)) { "bad sha256 for ${it.file}" }
                    require(it.bytes >= 0) { "bad size for ${it.file}" }
                }
            }
            require(arts.map { it.file }.toSet().size == arts.size) { "duplicate artifact files" }
            return BundleManifest(
                version, m.getInt("min_app_version"), m.getInt("tool_schema_version"),
                m.getJSONArray("tiers").let { t -> List(t.length()) { t.getString(it) } }, arts,
            )
        }

        /** "id=base64;id2=base64" (BuildConfig.BUNDLE_PUBLIC_KEYS) -> key id -> raw key. */
        fun keys(spec: String): Map<String, ByteArray> = spec.split(';').filter { it.contains('=') }.associate {
            val (id, b64) = it.split('=', limit = 2)
            id to Base64.getDecoder().decode(b64)
        }
    }
}
