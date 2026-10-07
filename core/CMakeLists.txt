cmake_minimum_required(VERSION 3.22.1)
project(farm_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FARM_CORE_BUILD_TESTS "Build host unit tests" OFF)
# Path to an extracted LiteRT 1.x AAR (headers/ and jni/<abi>/). Set by the
# Android build; when empty, the engine (inference) is not built.
set(FARM_CORE_LITERT_DIR "" CACHE PATH "Extracted LiteRT AAR directory")

# Portable part: image ops, quality gate, preprocessing, decision rules.
add_library(farm_core STATIC
  src/core.cpp
  src/image.cpp
  src/quality.cpp
  src/preprocess.cpp
  src/decision.cpp)
target_include_directories(farm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(farm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  target_compile_options(farm_core PRIVATE -Wall -Wextra -O2)
endif()

if(FARM_CORE_LITERT_DIR)
  add_library(litert SHARED IMPORTED)
  set_target_properties(litert PROPERTIES
    IMPORTED_LOCATION ${FARM_CORE_LITERT_DIR}/jni/${ANDROID_ABI}/libtensorflowlite_jni.so
    INTERFACE_INCLUDE_DIRECTORIES "${FARM_CORE_LITERT_DIR}/headers;${FARM_CORE_LITERT_DIR}/headers/external/org_tensorflow")
  add_library(litert_gpu SHARED IMPORTED)
  set_target_properties(litert_gpu PROPERTIES
    IMPORTED_LOCATION ${FARM_CORE_LITERT_DIR}/jni/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so)

  add_library(farm_engine STATIC src/engine.cpp)
  target_link_libraries(farm_engine PUBLIC farm_core litert litert_gpu)
  set_target_properties(farm_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_options(farm_engine PRIVATE -Wall -Wextra -O2)
endif()

if(FARM_CORE_BUILD_TESTS)
  enable_testing()
  add_executable(farm_core_tests tests/core_test.cpp tests/vision_test.cpp)
  target_link_libraries(farm_core_tests PRIVATE farm_core)
  target_compile_definitions(farm_core_tests PRIVATE FARM_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")
  add_test(NAME farm_core_tests COMMAND farm_core_tests)
endif()
