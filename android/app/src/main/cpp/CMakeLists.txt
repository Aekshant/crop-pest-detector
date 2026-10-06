cmake_minimum_required(VERSION 3.22.1)
project(farm_jni LANGUAGES CXX)

# Shared C++ core lives at the repo root.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core ${CMAKE_BINARY_DIR}/core)

add_library(farm_jni SHARED farm_jni.cpp)
target_link_libraries(farm_jni PRIVATE farm_core log)
