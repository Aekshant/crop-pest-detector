cmake_minimum_required(VERSION 3.22.1)
project(farm_jni LANGUAGES CXX)

# asan build type: sanitize our code (core + JNI). Prebuilt LiteRT is not instrumented.
if(FARM_SANITIZE STREQUAL "address")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
endif()

# Shared C++ core lives at the repo root; FARM_CORE_LITERT_DIR (set by Gradle)
# enables the LiteRT engine.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core ${CMAKE_BINARY_DIR}/core)

add_library(farm_jni SHARED farm_jni.cpp)
target_link_libraries(farm_jni PRIVATE farm_engine log)
target_compile_options(farm_jni PRIVATE -Wall -Wextra)
