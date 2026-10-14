package com.farm.app.vision

import android.content.Context
import java.io.File
import java.io.InputStream

/**
 * Where the vision model and its config come from: the APK's assets (the factory
 * bundle) or an activated / candidate model bundle directory. Both use the same
 * names: manifest.json, labels_v1.json, vision_fp16.tflite, vision_int8.tflite.
 */
fun interface VisionFiles {
    fun open(name: String): InputStream

    companion object {
        fun assets(context: Context) = VisionFiles { context.assets.open("vision/$it") }
        fun dir(dir: File) = VisionFiles { File(dir, it).inputStream() }
    }
}
