cmake_minimum_required(VERSION 3.22)

project(mlir-heir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(MLIRHeir)

add_subdirectory(tools)
add_subdirectory(tests)
