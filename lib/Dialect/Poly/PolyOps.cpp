#include "lib/Dialect/Poly/PolyOps.h"

namespace mlir {
  namespace heir {
    namespace poly {

      OpFoldResult PolyConstantOp::fold(PolyConstantOp::FoldAdaptor adaptor) {
        return adaptor.getCoeffecients();
      }
    } //namespace poly
  } // namespace heir
} // namespace mlir