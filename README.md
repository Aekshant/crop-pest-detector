# farm

Monorepo: app, native core, models and evals built in one CI.

| Dir        | What                                              |
|------------|---------------------------------------------------|
| `android/` | Kotlin Android app (Gradle). Links `core/` via NDK/CMake. |
| `core/`    | Portable C++ library. Builds for Android (NDK) and host (tests). |
| `ml/`      | Model training / export scripts.                  |
| `evals/`   | Evaluation harness and datasets. See [`evals/vision/`](evals/vision/README.md) (field + OOD sets). |
| `kb/`      | Knowledge base content.                           |
| `server/`  | Backend service.                                  |
| `docs/`    | Design notes. [`budget.md`](docs/budget.md) = pass/fail performance budget. |

## Build

```bash
# Android app + C++ core together (NDK build)
cd android && ./gradlew assembleDebug

# core on host, with tests
cmake -S core -B build/core -DFARM_CORE_BUILD_TESTS=ON && cmake --build build/core && ctest --test-dir build/core
```

Requires JDK 17+, Android SDK (platform 35), NDK 27.2.12479018, CMake 3.22.1.

## Android app

```bash
cd android && ./gradlew assembleDebug testDebugUnitTest
```

Kotlin + Jetpack Compose (Material 3), one activity, `navigation-compose` with four tabs:
Scan, Ask Laya, History, and Status (the design doc's debug screen). Each screen has a
`ViewModel` that exposes a `StateFlow` and does its work in `viewModelScope` coroutines.
UI strings are in English and Hindi (`values-hi`). Dependency versions are in
`android/gradle/libs.versions.toml`.
