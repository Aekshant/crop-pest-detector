package com.farm.app.bundle

import android.content.Context
import com.farm.app.kb.EmbeddingModel
import com.farm.app.kb.KbPack
import com.farm.app.laya.Templates
import com.farm.app.laya.ToolSchema
import com.farm.app.vision.VisionEngine
import com.farm.app.vision.VisionFiles
import com.farm.app.vision.resultOrNull
import org.json.JSONObject
import java.io.File

/**
 * The design doc's smoke test, run on a candidate bundle before it goes live:
 *  - every artifact this app needs is there; config files parse; the tool schema
 *    version matches the manifest
 *  - vision: the app's own golden images (assets/smoke, so a bundle can't bring its own
 *    answers) must get their expected labels, on CPU INT8 and, if shipped, GPU FP16
 *  - KB: the pack opens with treatment rows and chunks; the embedder makes a unit vector
 *  - Laya SLM: only a GGUF header check for now; the "3 golden prompts -> valid tool
 *    JSON" step needs the on-device SLM runtime, which the app doesn't have yet
 */
class BundleSmoke(private val context: Context, private val tier: String) : SmokeTest {
    override fun check(dir: File, manifest: BundleManifest): String? {
        val files = manifest.artifactsFor(tier).associate { it.name to File(dir, it.file) }
        val missing = REQUIRED.filter { it !in files }
        if (missing.isNotEmpty()) return "missing artifacts $missing"

        val schema = runCatching { ToolSchema.parse(files.getValue("tool_schema").readText()) }.getOrElse { return "tool schema: ${it.message}" }
        if (schema.version != manifest.toolSchemaVersion) return "tool schema file is v${schema.version}, manifest says v${manifest.toolSchemaVersion}"
        runCatching { Templates.parse(files.getValue("templates").readText()) }.onFailure { return "templates: ${it.message}" }
        runCatching { JSONObject(files.getValue("safety").readText()) }.onFailure { return "safety lexicon: ${it.message}" }

        vision(dir, gpu = false)?.let { return "vision CPU INT8: $it" }
        if ("vision_gpu" in files && !VisionEngine.isEmulator) vision(dir, gpu = true)?.let { return "vision GPU FP16: $it" }

        val pack = runCatching { KbPack.open(files.getValue("kb")) }.getOrElse { return "KB pack: ${it.message}" }
        if (pack.chunks.isEmpty() || pack.treatments.isEmpty()) return "KB pack is empty"
        runCatching {
            EmbeddingModel(files.getValue("kb_embedder"), files.getValue("kb_vocab")).use { m ->
                val v = m.embed("tomato leaf with brown spots")
                val norm = v.sumOf { (it * it).toDouble() }
                check(v.size == pack.meta["embedding_dim"]?.toInt() && norm in 0.99..1.01) { "bad embedding" }
            }
        }.onFailure { return "KB embedder: ${it.message}" }

        files["laya"]?.let { gguf ->
            val magic = gguf.inputStream().use { it.readNBytes(4) }
            if (!magic.contentEquals("GGUF".toByteArray())) return "laya model is not GGUF"
        }
        return null
    }

    private fun vision(dir: File, gpu: Boolean): String? {
        val golden = JSONObject(context.assets.open("smoke/golden.json").bufferedReader().use { it.readText() }).getJSONArray("images")
        return runCatching {
            VisionEngine.create(context, allowGpu = gpu, files = VisionFiles.dir(dir), pinToFastCores = false, perfHint = false).use { e ->
                var checked = 0
                for (i in 0 until golden.length()) {
                    val g = golden.getJSONObject(i)
                    val rgb = runCatching { context.assets.open("smoke/${g.getString("name")}.rgb").use { it.readBytes() } }.getOrNull() ?: continue
                    val r = e.analyzeRgb(rgb, g.getInt("width"), g.getInt("height")).resultOrNull() ?: return@use "golden ${g.getString("name")} failed"
                    // The expected label by ID, from the app's own golden.json: never looked up through
                    // the candidate's config, or a config with shuffled labels would agree with itself.
                    val want = g.getString("label")
                    if (r.top?.id != want) return@use "golden ${g.getString("name")}: got ${r.top?.id}, expected $want"
                    checked++
                }
                if (checked < MIN_GOLDENS) "only $checked golden images" else null
            }
        }.getOrElse { "could not load: ${it.message}" }
    }

    companion object {
        val REQUIRED = listOf("vision", "vision_config", "labels", "tool_schema", "templates", "safety", "kb", "kb_embedder", "kb_vocab")
        const val MIN_GOLDENS = 5
    }
}
