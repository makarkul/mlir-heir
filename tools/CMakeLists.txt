get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
    ${dialect_libs}
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
                                       