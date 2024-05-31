configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      MAIN_CONFIG
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    )

set (MLIR_HEIR_TEST_DEPENDS 
      FileCheck count not
      mlir-opt
      mlir-cpu-runner
      heir-opt
    )

  add_lit_testsuite(check-mlir-heir "Running the MLIR Heir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS 
      ${MLIR_HEIR_TEST_DEPENDS}
)
