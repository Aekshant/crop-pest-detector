#include <jni.h>

#include "farm/core.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_farm_app_NativeCore_version(JNIEnv* env, jobject /* this */) {
  return env->NewStringUTF(farm::version().c_str());
}
