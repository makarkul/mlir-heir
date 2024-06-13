#ifndef LIB_DIALECT_POLY_POLYOPS_TD_
#define LIB_DIALECT_POLY_POLYOPS_TD_

include "PolyDialect.td"
include "PolyTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Type constraints for poly binop arguments: polys, vectors of polys or
// tensor of polys.
def PolyOrContainer : TypeOrContainer<Polynomial, "poly-or-container">;

class PolyBinOp<string mnemonic> : Op<PolyDialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
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

def PolyFromTensorOp : Op<PolyDialect, "from_tensor", [Pure]> {
  let summary = "Creates a Polynomial from integer coeffecients stored in a tensor";
  let arguments = (ins TensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";
  let hasFolder = 1;
}

def PolyEvalOp : Op<PolyDialect, "eval", [AllTypesMatch<["point", "output"]>]> {
  let summary = "Evaluates a Polynomial at a given input value";
  let arguments = (ins Polynomial:$input, AnyInteger:$point);
  let results = (outs AnyInteger:$output);
  let assemblyFormat = "$input `,` $point attr-dict `:` `(` qualified(type($input)) `,` type($point) `)` `->` type($output)";
}

def PolyConstantOp : Op<PolyDialect, "constant", [Pure, ConstantLike]> {
  let summary = "Define a constant polynomial via an attribute";
  let arguments = (ins AnyIntElementsAttr:$coeffecients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coeffecients attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
}
#endif//LIB_DIALECT_POLY_POLYOPS_TD_