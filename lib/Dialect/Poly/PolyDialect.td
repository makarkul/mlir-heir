#ifndef LIB_DIALECT_POLY_POLYDIALECT_TD_
#define LIB_DIALECT_POLY_POLYDIALECT_TD_

include "mlir/IR/DialectBase.td"

def PolyDialect : Dialect {
  let name = "poly";
  let summary = "A dialect for polynomial math";
  let description = [{
    The polynomial dialect defines types and operations for single-variable
    polynomials over integers.
  }];

  let cppNamespace = "::mlir::heir::poly";

  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

#endif//LIB_DIALECT_POLY_POLYDIALECT_TD_
