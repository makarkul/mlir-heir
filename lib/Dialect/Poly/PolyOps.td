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

// Inject verification that all integer-like arguments are 32-bits
def Has32BitArguments : NativeOpTrait<"Has32BitArguments"> {
  let cppNamespace = "::mlir::heir::poly";
}

class PolyBinOp<string mnemonic> : Op<PolyDialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
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

def PolyToTensorOp : Op<PolyDialect, "to_tensor", [Pure]> {
  let summary = "Converts a polynomial to a tensor of it's integer coeffecients";
  let arguments = (ins Polynomial:$input);
  let results = (outs TensorOf<[AnyInteger]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";
}

def IntOrComplex : AnyTypeOf<[AnyInteger, AnyComplex]>;

def PolyEvalOp : Op<PolyDialect, "eval", [AllTypesMatch<["point", "output"]>, Has32BitArguments]> {
  let summary = "Evaluates a Polynomial at a given input value";
  let arguments = (ins Polynomial:$polynomial, IntOrComplex:$point);
  let results = (outs IntOrComplex:$output);
  let assemblyFormat = "$polynomial `,` $point attr-dict `:` `(` qualified(type($polynomial)) `,` type($point) `)` `->` type($output)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def PolyConstantOp : Op<PolyDialect, "constant", [Pure, ConstantLike]> {
  let summary = "Define a constant polynomial via an attribute";
  let arguments = (ins AnyIntElementsAttr:$coeffecients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coeffecients attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
}
#endif//LIB_DIALECT_POLY_POLYOPS_TD_
