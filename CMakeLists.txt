cmake_minimum_required(VERSION 3.22)

project(mlir-heir LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(MLIRHeir)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    )


add_llvm_executable(heir-opt
  tools/heir-opt.cpp
  )

target_link_libraries(heir-opt
  PRIVATE
    ${LIBS}
  )

add_subdirectory(tests)
