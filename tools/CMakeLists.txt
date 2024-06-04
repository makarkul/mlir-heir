get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    AffineFullUnroll
    MulToAdd
    MLIROptLib
    )

add_llvm_executable(heir-opt
  heir-opt.cpp
  )

target_link_libraries(heir-opt
  PRIVATE
    ${LIBS}
  )

mlir_check_all_link_libraries(heir-opt)
