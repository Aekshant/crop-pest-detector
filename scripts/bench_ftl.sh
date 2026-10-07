#!/usr/bin/env bash
# S4: run the benchmark on a phone in Firebase Test Lab (budget 4 GB device).
#
# One-time setup (needs your Google account; the Spark plan includes a few
# free physical-device runs per day):
#   gcloud auth login
#   gcloud config set project <your-firebase-project-id>
#   gcloud services enable testing.googleapis.com toolresults.googleapis.com
#
# Pick a device (look for ~4 GB RAM budget phones, e.g. Samsung Galaxy A-series):
#   gcloud firebase test android models list --filter="form=PHYSICAL"
#
#   MODEL=<model-id> VERSION=<api-level> scripts/bench_ftl.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL:?set MODEL, see: gcloud firebase test android models list --filter=form=PHYSICAL}"
: "${VERSION:?set VERSION (API level supported by that model)}"
RESULTS="farmlens-s4/$(date +%Y%m%d-%H%M%S)-${MODEL}"

(cd android && ./gradlew -q assembleDebug assembleDebugAndroidTest)
gcloud firebase test android run \
  --type instrumentation \
  --app android/app/build/outputs/apk/debug/app-debug.apk \
  --test android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk \
  --device "model=${MODEL},version=${VERSION},locale=en,orientation=portrait" \
  --test-targets "class com.farm.app.vision.GoldenTest" \
  --test-targets "class com.farm.app.vision.DeviceBenchmark" \
  --directories-to-pull /sdcard/Android/data/com.farm.app/files \
  --results-dir "$RESULTS" \
  --timeout 20m

# Results land in the Test Lab bucket printed above; copy the report back:
echo
echo "Copy the report with:"
echo "  gsutil cp 'gs://<test-lab-bucket>/${RESULTS}/**/bench.json' docs/devices/${MODEL}.json"
echo "  python3 scripts/bench_summary.py docs/devices/*.json"
