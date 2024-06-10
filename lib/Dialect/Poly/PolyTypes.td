#ifndef LIB_DIALECT_POLY_POLYTYPES_TD_
#define LIB_DIALECT_POLY_POLYTYPES_TD_

include "PolyDialect.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class PolyType<string name, string typeMnemonic> : TypeDef<PolyDialect, name> {
  let mnemonic = typeMnemonic;
}

def PolyNomial : PolyType<"Polynomial", "poly"> {
  let summary = "A polynomial with u32 coffecients";

  let description = [{
    A type for polynomials with integer coeffecients in a single-variable polynomial ring.
  }];

  let parameters = (ins "int":$degreeBound);
  let assemblyFormat = "`<` $degreeBound `>`";
}

#endif//LIB_DIALECT_POLY_POLYTYPES_TD_