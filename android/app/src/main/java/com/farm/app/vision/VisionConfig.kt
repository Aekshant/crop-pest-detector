package com.farm.app.vision

import android.content.Context
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class Label(val id: String, val nameEn: String, val nameHi: String, val kind: String) {
    /** Diseases get a lesion-area severity estimate; pests, healthy and unknown do not. */
    val isDisease: Boolean get() = kind == "disease"
}

/** Per-model-file calibration from the manifest (quantization shifts probabilities). */
data class ModelCalibration(val temperature: Float, val rejectBelow: Float)

/** Vision settings shipped in assets/vision (ml/bundle/manifest_draft.json + ml/labels_v1.json). */
data class VisionConfig(
    val modelVersion: String,
    val labels: List<Label>,
    val unknownIndex: Int,
    val inputSize: Int,
    val confidentAt: Float,
    val fp16: ModelCalibration,
    val int8: ModelCalibration,
    val minSharpness: Float,
    val minBrightness: Float,
    val maxBrightness: Float,
) {
    companion object {
        private const val DIR = "vision"

        fun load(context: Context): VisionConfig {
            val v = JSONObject(readText(context, "$DIR/manifest.json")).getJSONObject("vision")
            val ids = v.getJSONArray("labels").let { a -> List(a.length()) { a.getString(it) } }
            val names = JSONObject(readText(context, "$DIR/labels_v1.json")).getJSONArray("labels")
            val byId = (0 until names.length()).map { names.getJSONObject(it) }.associateBy { it.getString("id") }
            val labels = ids.map { id ->
                val n = byId.getValue(id)
                Label(id, n.getString("name_en"), n.getString("name_hi"), n.getString("kind"))
            }
            val files = v.getJSONObject("files")
            fun cal(key: String) = files.getJSONObject(key).let {
                ModelCalibration(it.getDouble("temperature").toFloat(), it.getDouble("reject_below").toFloat())
            }
            val gate = v.getJSONObject("quality_gate")
            return VisionConfig(
                modelVersion = v.getString("model_version"),
                labels = labels,
                unknownIndex = ids.indexOf("unknown"),
                inputSize = v.getJSONObject("input").getInt("size"),
                confidentAt = files.getJSONObject("int8").getDouble("confident_at_or_above").toFloat(),
                fp16 = cal("fp16"),
                int8 = cal("int8"),
                minSharpness = gate.getDouble("min_sharpness").toFloat(),
                minBrightness = gate.getDouble("min_brightness").toFloat(),
                maxBrightness = gate.getDouble("max_brightness").toFloat(),
            )
        }

        fun modelBuffer(context: Context, name: String): ByteBuffer {
            val bytes = context.assets.open("$DIR/$name").use { it.readBytes() }
            return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes).apply { rewind() }
        }

        private fun readText(context: Context, path: String) =
            context.assets.open(path).bufferedReader().use { it.readText() }
    }
}
