#ifndef LIB_DIALECT_POLY_POLYOPS_TD_
#define LIB_DIALECT_POLY_POLYOPS_TD_

include "PolyDialect.td"
include "PolyTypes.td"
include "mlir/IR/OpBase.td"

class PolyBinOp<string mnemonic> : Op<PolyDialect, mnemonic> {
  let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}

def PolyAddOp : PolyBinOp<"add"> {
  let summary = "Addition operation between  polynomials";
}

def PolySubOp : PolyBinOp<"sub"> {
  let summary = "Subtraction operation between polynomials";
}

def PolyMulOp : PolyBinOp<"mul"> {
  let summary = "Multiplication operation between polynomials";
}

#endif//LIB_DIALECT_POLY_POLYOPS_TD_