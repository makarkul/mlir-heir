#ifndef LIB_DIALECT_POLY_POLYPATTERNS_TD_
#define LIB_DIALECT_POLY_POLYPATTERNS_TD_

include "PolyOps.td"
include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"

def LiftConjThroughEval : Pat<
    (PolyEvalOp $f, (ConjOp $z, $fastmath)),
    (ConjOp (PolyEvalOp $f, $z), $fastmath)
>;

#endif//LIB_DIALECT_POLY_POLYPATTERNS_TD_
