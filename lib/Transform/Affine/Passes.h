#ifndef LIB_TRANSFORM_AFFINE_PASSES_H
#define LIB_TRANSFORM_AFFINE_PASSES_H

#include "lib/Transform/Affine/AffineFullUnroll.h"
#include "lib/Transform/Affine/AffineFullUnrollPatternRewrite.h"

namespace mlir {
  namespace heir {

#define GEN_PASS_REGISTRATION
#include "lib/Transform/Affine/Passes.h.inc"

  }
}
#endif//LIB_TRANSFORM_AFFINE_PASSES_H