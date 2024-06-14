#ifndef LIB_DIALECT_POLY_POLYPATTERNS_TD_
#define LIB_DIALECT_POLY_POLYPATTERNS_TD_

include "PolyOps.td"
include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"

def LiftConjThroughEval : Pat<
    (PolyEvalOp $f, (ConjOp $z, $fastmath)),
    (ConjOp (PolyEvalOp $f, $z), $fastmath)
>;

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses
def DifferenceOfSquares : Pattern<
  (PolySubOp (PolyMulOp:$lhs $x, $x), (PolyMulOp:$rhs $y, $y)),
  [
    (PolyAddOp:$sum $x, $y),
    (PolySubOp:$diff $x, $y),
    (PolyMulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;

#endif//LIB_DIALECT_POLY_POLYPATTERNS_TD_
