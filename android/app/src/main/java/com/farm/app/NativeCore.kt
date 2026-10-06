package com.farm.app

object NativeCore {
    init {
        System.loadLibrary("farm_jni")
    }

    external fun version(): String
}
