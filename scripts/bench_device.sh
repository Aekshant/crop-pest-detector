#!/usr/bin/env bash
# S4: run the on-device benchmark + golden tests on a phone connected over USB.
#
#   1. On the phone: Settings > About phone > tap "Build number" 7 times,
#      then Developer options > USB debugging = on. Connect USB, accept the prompt.
#   2. scripts/bench_device.sh            (or: scripts/bench_device.sh <adb-serial>)
#
# Writes docs/devices/<model>.json and prints a summary against the budget.
set -euo pipefail
cd "$(dirname "$0")/.."
ADB=(adb)
command -v adb >/dev/null || ADB=("$HOME/Android/Sdk/platform-tools/adb")
[ $# -ge 1 ] && ADB+=(-s "$1")

(cd android && ./gradlew -q assembleDebug assembleDebugAndroidTest)
"${ADB[@]}" install -r -g android/app/build/outputs/apk/debug/app-debug.apk
"${ADB[@]}" install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk

echo "Running golden test + benchmark (a few minutes)..."
"${ADB[@]}" shell am instrument -w \
  -e class com.farm.app.vision.GoldenTest,com.farm.app.vision.DeviceBenchmark \
  com.farm.app.test/androidx.test.runner.AndroidJUnitRunner | tail -5

model=$("${ADB[@]}" shell getprop ro.product.model | tr -d '\r' | tr -c 'A-Za-z0-9_-' '_' | sed 's/_*$//')
out="docs/devices/${model}.json"
"${ADB[@]}" exec-out run-as com.farm.app cat files/bench.json > "$out"
echo "report: $out"
python3 scripts/bench_summary.py docs/devices/*.json
