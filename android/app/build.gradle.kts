plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.compose.compiler)
}

val repoRoot = rootProject.projectDir.parentFile

// LiteRT 1.x AARs ship the TFLite C API headers and .so files. The AARs are app
// dependencies (so the .so files are packaged); here they are also unpacked so
// the C++ core can compile and link against them.
val litertNative: Configuration by configurations.creating { isTransitive = false }
val litertDir = layout.buildDirectory.dir("litert")
val extractLiteRt by tasks.registering(Sync::class) {
    from(litertNative.elements.map { files -> files.map { zipTree(it.asFile) } }) {
        include("jni/**", "headers/**")
    }
    into(litertDir)
}

// Model files and manifests the app loads at startup (Git LFS in ml/).
val visionAssets = layout.buildDirectory.dir("generated/visionAssets")
val copyVisionAssets by tasks.registering(Sync::class) {
    from(repoRoot.resolve("ml/runs/v3b_lite0_cpu/export")) { include("vision_fp16.tflite", "vision_int8.tflite") }
    from(repoRoot.resolve("ml/bundle/manifest_draft.json")) { rename { "manifest.json" } }
    from(repoRoot.resolve("ml/labels_v1.json"))
    into(visionAssets.map { it.dir("vision") })
}

android {
    namespace = "com.farm.app"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.farm.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                arguments += "-DFARM_CORE_LITERT_DIR=${litertDir.get().asFile.absolutePath}"
            }
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    sourceSets {
        getByName("main").assets.directories += visionAssets.get().asFile.absolutePath
        // Golden images + Python reference outputs for the on-device test.
        getByName("androidTest").assets.directories += repoRoot.resolve("core/tests/golden").absolutePath
    }

    androidResources {
        noCompress += "tflite"  // memory-mapped / read directly
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

tasks.named("preBuild") { dependsOn(extractLiteRt, copyVisionAssets) }
tasks.configureEach {
    if (name.startsWith("configureCMake") || name.startsWith("buildCMake")) dependsOn(extractLiteRt)
    if (name.contains("Assets", ignoreCase = true) && name.startsWith("merge")) dependsOn(copyVisionAssets)
}

dependencies {
    litertNative("com.google.ai.edge.litert:litert:${libs.versions.litert.get()}@aar")
    litertNative("com.google.ai.edge.litert:litert-gpu:${libs.versions.litert.get()}@aar")

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.navigation.compose)
    implementation(libs.coroutines.android)
    implementation(libs.litert.asProvider())
    implementation(libs.litert.gpu)
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)
    debugImplementation(libs.compose.ui.tooling)

    testImplementation(libs.junit)
    testImplementation(libs.coroutines.test)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.junit)
}
