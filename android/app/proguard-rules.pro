# Release shrinking (R8). Native methods keep their names via proguard-android-optimize.txt;
# ONNX Runtime and LiteRT ship their own consumer rules.

# JNI entry points are bound by name: Java_com_farm_app_vision_VisionNative_native*.
-keep class com.farm.app.vision.VisionNative { native <methods>; }
-keep class com.farm.app.NativeCore { native <methods>; }

# WorkManager (bundle updates) stores its jobs in Room, which creates the generated
# WorkDatabase_Impl by reflection. Full-mode R8 removed its constructor: crash at startup.
-keep class * extends androidx.room.RoomDatabase { <init>(); }
