#ifndef LIB_TRANSFORM_ARITH_PASSES_H_
#define LIB_TRANSFORM_ARITH_PASSES_H_

#include "lib/Transform/Arith/MulToAdd.h"

namespace mlir {
  namespace heir {

#define GEN_PASS_REGISTRATION
#include "lib/Transform/Arith/Passes.h.inc"

  }
}

#endif//LIB_TRANSFORM_ARITH_PASSES_H_