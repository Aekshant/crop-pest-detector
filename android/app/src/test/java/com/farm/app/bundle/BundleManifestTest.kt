package com.farm.app.bundle

import org.bouncycastle.crypto.params.Ed25519PrivateKeyParameters
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.security.SecureRandom
import java.util.Base64

class BundleManifestTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun acceptsAManifestSignedByThePythonBuilder() {
        val f = JSONObject(javaClass.getResource("/bundle_fixture.json")!!.readText())
        val keys = mapOf("python-test" to Base64.getDecoder().decode(f.getString("public_key")))
        val m = BundleManifest.verify(f.getJSONObject("envelope").toString().toByteArray(), keys)
        assertEquals("2026.01.01", m.version)
        assertEquals(listOf("labels_v1.json"), m.artifactsFor("C").map { it.file })
        assertEquals(listOf("labels_v1.json", "laya.gguf"), m.artifactsFor("A").map { it.file })
    }

    @Test fun rejectsTamperedUnknownAndWrongKeys() {
        val kit = BundleTestKit(tmp.root)
        try {
            val good = """{"bundle_version":"1","min_app_version":1,"tool_schema_version":1,"tiers":["A"],"artifacts":[]}""".toByteArray()
            BundleManifest.verify(kit.envelope(good).toByteArray(), kit.trusted)  // sanity: accepted

            val env = JSONObject(kit.envelope(good))
            val tampered = good.copyOf().also { it[2] = 'B'.code.toByte() }
            env.put("manifest", Base64.getEncoder().encodeToString(tampered))
            assertThrows(BundleManifest.BadSignature::class.java) { BundleManifest.verify(env.toString().toByteArray(), kit.trusted) }

            assertThrows(BundleManifest.BadSignature::class.java) {
                BundleManifest.verify(kit.envelope(good, id = "someone-else").toByteArray(), kit.trusted)
            }
            val stranger = Ed25519PrivateKeyParameters(SecureRandom())
            assertThrows(BundleManifest.BadSignature::class.java) {
                BundleManifest.verify(kit.envelope(good, signer = stranger).toByteArray(), kit.trusted)
            }
            assertThrows(BundleManifest.BadSignature::class.java) { BundleManifest.verify("not json".toByteArray(), kit.trusted) }
        } finally {
            kit.close()
        }
    }

    @Test fun refusesUnsafeFileNamesEvenWhenSigned() {
        for (bad in listOf("../evil.so", "a/b", "", "manifest.json.env")) {
            val m = """{"bundle_version":"1","min_app_version":1,"tool_schema_version":1,"tiers":["A"],
                "artifacts":[{"name":"x","file":"$bad","sha256":"${"0".repeat(64)}","bytes":1}]}""".toByteArray()
            assertThrows(bad, IllegalArgumentException::class.java) { BundleManifest.parse(m) }
        }
    }

    @Test fun keysFromBuildConfig() {
        val k = BundleManifest.keys("a=AAAA;b=AQID")
        assertEquals(setOf("a", "b"), k.keys)
        assertEquals(listOf<Byte>(1, 2, 3), k.getValue("b").toList())
    }
}
