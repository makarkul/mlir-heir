# Inlining `add_mlir_dialect(Poly poly)` commands so hat 
# we can custom name `*.inc` generated files
set(LLVM_TARGET_DEFINITIONS PolyDialect.td)
mlir_tablegen(PolyDialect.h.inc -gen-dialect-decls -dialect=poly)
mlir_tablegen(PolyDialect.cpp.inc -gen-dialect-defs -dialect=poly)

add_public_tablegen_target(MLIRPolyOpsIncGen)

add_dependencies(mlir-headers MLIRPolyOpsIncGen)

add_mlir_dialect_library(MLIRPoly
  PolyDialect.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly

  LINK_LIBS
    PUBLIC
  )
