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
}