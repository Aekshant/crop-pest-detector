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

// AddressSanitizer build (A8): NDK ASan runtime + the NDK's official wrap.sh,
// copied from the installed NDK at build time (nothing binary in git).
val asanFiles = layout.buildDirectory.dir("generated/asan")
val copyAsanRuntime by tasks.registering(Sync::class) {
    val ndk = androidComponents.sdkComponents.ndkDirectory.get().asFile
    val clangLib = ndk.resolve("toolchains/llvm/prebuilt/linux-x86_64/lib/clang")
        .listFiles()?.maxByOrNull { it.name }?.resolve("lib/linux")
    for ((abi, arch) in mapOf("arm64-v8a" to "aarch64", "x86_64" to "x86_64")) {
        from(clangLib!!.resolve("libclang_rt.asan-$arch-android.so")) { into("jniLibs/$abi") }
        from(ndk.resolve("wrap.sh/asan.sh")) {
            into("resources/lib/$abi")
            rename { "wrap.sh" }
        }
    }
    into(asanFiles)
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

        externalNativeBuild {
            cmake {
                arguments += "-DFARM_CORE_LITERT_DIR=${litertDir.get().asFile.absolutePath}"
            }
        }
    }

    buildTypes {
        getByName("debug") {
            ndk { abiFilters += listOf("arm64-v8a", "x86_64") }  // x86_64 for the emulator
        }
        // Phones only (arm64-v8a): x86_64 alone is ~55 MB of native libraries (ONNX Runtime 41 MB).
        // Unsigned until a release keystore exists (signingConfig goes here).
        getByName("release") {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            ndk { abiFilters += listOf("arm64-v8a") }
            // Local device testing only: ./gradlew assembleRelease -Pfarm.signReleaseWithDebugKey
            if (providers.gradleProperty("farm.signReleaseWithDebugKey").isPresent) {
                signingConfig = signingConfigs.getByName("debug")
            }
        }
        // Debug build with AddressSanitizer on all our native code:
        //   ./gradlew assembleAsan
        //   ./gradlew assembleAsanAndroidTest -Pfarm.testBuildType=asan
        create("asan") {
            initWith(getByName("debug"))
            matchingFallbacks += "debug"
            externalNativeBuild {
                cmake {
                    arguments += listOf("-DANDROID_STL=c++_shared", "-DFARM_SANITIZE=address")
                }
            }
        }
    }
    testBuildType = providers.gradleProperty("farm.testBuildType").getOrElse("debug")

    buildFeatures {
        compose = true
        buildConfig = true
    }

    sourceSets {
        getByName("main").assets.directories += visionAssets.get().asFile.absolutePath
        getByName("asan").jniLibs.directories += asanFiles.get().dir("jniLibs").asFile.absolutePath
        getByName("asan").resources.directories += asanFiles.get().dir("resources").asFile.absolutePath
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

tasks.named("preBuild") { dependsOn(extractLiteRt, copyVisionAssets, copyAsanRuntime) }

// wrap.sh only runs when native libs are extracted to disk.
androidComponents {
    onVariants(selector().withBuildType("asan")) { it.packaging.jniLibs.useLegacyPackaging.set(true) }
}
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
    implementation(libs.onnxruntime.android)  // KB question embeddings (e5-small int8)
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)
    debugImplementation(libs.compose.ui.tooling)

    testImplementation(libs.junit)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.org.json)  // real org.json on the JVM (android.jar only has stubs)
    testImplementation(libs.sqlite.jdbc)  // reads the KB pack in KbRetrieverTest
    testImplementation(libs.onnxruntime.jvm)  // desktop ONNX Runtime for EmbeddingModelTest
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.junit)
}

// Unit tests run on the desktop JVM: use the desktop ONNX Runtime (same ai.onnxruntime API) instead of the Android AAR.
configurations.matching { it.name.endsWith("UnitTestRuntimeClasspath") }.configureEach {
    exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime-android")
}

// Unit tests read repo data at run time (bundle JSON, evals, KB CSVs, kb/dist). Declare it as input,
// or Gradle reports stale results as up to date after a data-only change.
tasks.withType<Test>().configureEach {
    val repo = rootProject.projectDir.parentFile
    inputs.files(fileTree(repo) {
        include("ml/bundle/**", "ml/labels_v1.json", "evals/laya/**", "evals/kb/**", "kb/*.csv", "kb/dist/**")
        exclude("**/__pycache__/**")
    }).withPathSensitivity(PathSensitivity.RELATIVE).withPropertyName("repoData")
}

// The merged manifest must not regain what libraries sneak in (ONNX Runtime telemetry, permissions
// implied by LiteRT GPU). Run in CI: ./gradlew checkMergedManifests
val forbiddenInManifest = listOf(
    "ai.onnxruntime.TelemetryInitializer",
    "android.permission.READ_PHONE_STATE",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WRITE_EXTERNAL_STORAGE",
)
tasks.register("checkMergedManifests") {
    dependsOn("processDebugManifest", "processReleaseManifest")
    val manifests = listOf("debug", "release").map {
        layout.buildDirectory.file("intermediates/merged_manifests/$it/process${it.replaceFirstChar(Char::uppercase)}Manifest/AndroidManifest.xml")
    }
    inputs.files(manifests)
    doLast {
        manifests.forEach { m ->
            val text = m.get().asFile.readText()
            val found = forbiddenInManifest.filter { text.contains(it) }
            check(found.isEmpty()) { "${m.get().asFile}: must not contain $found" }
        }
    }
}

