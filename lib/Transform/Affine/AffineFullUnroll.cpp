#include "lib/Transform/Affine/AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Pass/Pass.h"

using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

namespace mlir {
  namespace heir {

    void AffineFullUnrollPass::runOnOperation() {
      getOperation().walk([&](AffineForOp op) {
        if (failed(loopUnrollFull(op))) {
          op.emitError("unrolling failed");
          signalPassFailure();
        }
      });
    }
  } // namespace tutorial
} // namespace mlir
