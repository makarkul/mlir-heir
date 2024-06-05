#include "lib/Transform/Affine/AffineFullUnrollPatternRewrite.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace heir {
#define GEN_PASS_DEF_AFFINEFULLUNROLLPATTERNREWRITE
#include "lib/Transform/Affine/Passes.h.inc"

    using mlir::affine::AffineForOp;
    using mlir::affine::loopUnrollFull;

    // A pattern that matches on AffineForop and unrolls it
    struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp> {
      AffineFullUnrollPattern(mlir::MLIRContext *context)
        : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {}

      LogicalResult matchAndRewrite(AffineForOp op,
          PatternRewriter &rewriter) const override {
        // This is technically not allowed, since in a RewritePattern all
        // modifications to the IR are supposed to go through the `rewriter` arg
        // but itworks for our limited test cases.
        return loopUnrollFull(op);
      }
    };

    // A Pass that involes the pattern rewrite engine.
    struct AffineFullUnrollPatternRewrite
      : impl::AffineFullUnrollPatternRewriteBase<AffineFullUnrollPatternRewrite> {
        using AffineFullUnrollPatternRewriteBase::AffineFullUnrollPatternRewriteBase;
        void runOnOperation() {
          mlir::RewritePatternSet patterns(&getContext());
          patterns.add<AffineFullUnrollPattern>(&getContext());
          // One could use GreedyRewriteConfig here to slightly tweak the 
          // behaiour of the pattern application
          (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
        }
      };
  }// namespace heir
}//namespace mlir
