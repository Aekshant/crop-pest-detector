package com.farm.app.vision

import android.content.Context
import java.io.File
import java.io.InputStream

/**
 * Where the vision model and its config come from: the APK's assets (the factory
 * bundle) or an activated / candidate model bundle directory. Both use the same
 * names: manifest.json, labels_v1.json, vision_fp16.tflite, vision_int8.tflite.
 */
interface VisionFiles {
    fun open(name: String): InputStream
    fun has(name: String): Boolean

    companion object {
        fun assets(context: Context) = object : VisionFiles {
            override fun open(name: String) = context.assets.open("vision/$name")
            override fun has(name: String) = context.assets.list("vision")?.contains(name) == true
        }

        fun dir(dir: File) = object : VisionFiles {
            override fun open(name: String) = File(dir, name).inputStream()
            override fun has(name: String) = File(dir, name).isFile
        }
    }
}
