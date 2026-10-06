cmake_minimum_required(VERSION 3.22.1)
project(farm_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FARM_CORE_BUILD_TESTS "Build host unit tests" OFF)

add_library(farm_core STATIC src/core.cpp)
target_include_directories(farm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(farm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(FARM_CORE_BUILD_TESTS)
  enable_testing()
  add_executable(farm_core_tests tests/core_test.cpp)
  target_link_libraries(farm_core_tests PRIVATE farm_core)
  add_test(NAME farm_core_tests COMMAND farm_core_tests)
endif()
