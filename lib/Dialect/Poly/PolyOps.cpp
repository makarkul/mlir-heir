#include "lib/Dialect/Poly/PolyOps.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
  namespace heir {
    namespace poly {

      OpFoldResult PolyConstantOp::fold(PolyConstantOp::FoldAdaptor adaptor) {
        return adaptor.getCoeffecients();
      }

      OpFoldResult PolyAddOp::fold(PolyAddOp::FoldAdaptor adaptor) {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
          adaptor.getOperands(), [&](APInt a, APInt b) { 
            return a + b; 
          }
        );
      }

      OpFoldResult PolySubOp::fold(PolySubOp::FoldAdaptor adaptor) {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
          adaptor.getOperands(), [&](APInt a, APInt b) {
            return a - b;
          }
        );
      }

      OpFoldResult PolyMulOp::fold(PolyMulOp::FoldAdaptor adaptor) {
        auto lhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[0]);
        auto rhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[1]);

        if (!rhs || !lhs) {
          return nullptr;
        }

        auto degree = getResult().getType().cast<PolynomialType>().getDegreeBound();
        auto maxIndex = lhs.size() + rhs.size() - 1;

        SmallVector<APInt, 8> result;
        result.reserve(maxIndex);
        for (int i = 0; i < maxIndex; ++i) {
          result.push_back(APInt((*lhs.begin()).getBitWidth(), 0));
        }

        int i = 0;
        for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>();
            ++lhsIt) {
          int j = 0;
          for (auto rhsIt = rhs.value_begin<APInt>(); rhsIt != rhs.value_end<APInt>();
                ++rhsIt) {
            // Index is modulo degree because poly's semantics are defined modulo x^N = 1
            result[(i + j) % degree] += *rhsIt * (*lhsIt);
            ++j;
          }
          ++i;
        }
        return DenseIntElementsAttr::get(
          RankedTensorType::get(static_cast<int64_t>(result.size()),
                                IntegerType::get(getContext(), 32)),
          result);
      }

      OpFoldResult PolyFromTensorOp::fold(PolyFromTensorOp::FoldAdaptor adaptor) {
        // Returns null if the cast failed, which corresponds to a failed fold.
        return dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
      }
      
      LogicalResult PolyEvalOp::verify() {
        return getPoint().getType().isSignlessInteger(32)
                  ? success()
                  : emitOpError("argument point must be a 32-bit integer");
      }

      // Rewrites (x^2 - y^2) = (x+y)(x-y) if x^2 and y^2 have no other uses
      struct DifferenceOfSquares : public  OpRewritePattern<PolySubOp> {
        DifferenceOfSquares(mlir::MLIRContext *context)
          : OpRewritePattern<PolySubOp>(context, /*benifit=*/1) {}

        LogicalResult matchAndRewrite(PolySubOp op,
                                      PatternRewriter &rewriter) const override {
          Value lhs = op.getOperand(0);
          Value rhs = op.getOperand(1);

          // If either arg has another use, then this rewrite is probably less 
          // efficient, becuase it cannot delete the mul ops
          if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
            return failure();
          }

          auto rhsMul = rhs.getDefiningOp<PolyMulOp>();
          auto lhsMul = lhs.getDefiningOp<PolyMulOp>();
          if (!lhsMul || !rhsMul) {
            return failure();
          }

          bool rhsMulOpsAgree = rhsMul.getLhs() == rhsMul.getRhs();
          bool lhsMulOpsAgree = lhsMul.getLhs() == lhsMul.getRhs();

          if (!rhsMulOpsAgree || !lhsMulOpsAgree) {
            return failure();
          }

          auto x = lhsMul.getLhs();
          auto y = rhsMul.getLhs();

          PolyAddOp newAdd = rewriter.create<PolyAddOp>(op.getLoc(), x, y);
          PolySubOp newSub = rewriter.create<PolySubOp>(op.getLoc(), x, y);
          PolyMulOp newMul = rewriter.create<PolyMulOp>(op.getLoc(), newAdd, newSub);

          rewriter.replaceOp(op, newMul);
          // We don't need to remove the origin ops because MLIR already has
          // canonicalizaton patterns that remove unused ops.

          return success();
        }
      };

      void PolyAddOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
            ::mlir::MLIRContext *context) {}

      void PolySubOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
            ::mlir::MLIRContext *context) {

        results.add<DifferenceOfSquares>(context);
      }

      void PolyMulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
            ::mlir::MLIRContext *context) {}

    } //namespace poly
  } // namespace heir
} // namespace mlir