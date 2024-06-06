#ifndef LIB_TRANSFORM_ARITH_MULTTOADD_H_
#define LIB_TRANSFORM_ARITH_MULTTOADD_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace heir {

#define GEN_PASS_DECL_MULTOADD
#include "lib/Transform/Arith/Passes.h.inc"

  } // namespace heir
} // namespace mlir


#endif//LIB_TRANSFORM_ARITH_MULTTOADD_H_
