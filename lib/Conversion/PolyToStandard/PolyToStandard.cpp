#include "lib/Conversion/PolyToStandard/PolyToStandard.h"

#include "lib/Dialect/Poly/PolyOps.h"
#include "lib/Dialect/Poly/PolyTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
  namespace heir {
    namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "lib/Conversion/PolyToStandard/PolyToStandard.h.inc"

      class PolyToStandardTypeConvertor : public TypeConverter {
      public:
        PolyToStandardTypeConvertor(MLIRContext *ctx) {
          addConversion([](Type type) {return type; });
          addConversion([ctx](PolynomialType type) -> Type {
            int degreeBound = type.getDegreeBound();
            IntegerType elementTy = 
              IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signless);
            return RankedTensorType::get({degreeBound}, elementTy);
          });

          // We don't include any custom materialization hooks because this lowering 
          // is all done in a single pass. The dialect conversion framework works by 
          // resolving intermediate (mid-pass) type conflicts by inserting 
          // unrealized_conversion_cast ops, and only converting those to custom
          // materializations if they persis at the end of the pass. In our case,
          // we'd only need to use custom materializations if we split this lowering
          // across multiple passes.
        }
      };

      struct ConvertAdd : public OpConversionPattern<PolyAddOp> {
        ConvertAdd(mlir::MLIRContext *context) 
          : OpConversionPattern<PolyAddOp>(context) {}

        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(
          PolyAddOp op, OpAdaptor adaptor,
          ConversionPatternRewriter &rewriter) const override {
            arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
              op.getLoc(), adaptor.getLhs(), adaptor.getRhs());

            rewriter.replaceOp(op.getOperation(), addOp);
            return success();
          }
      };

      struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
        using PolyToStandardBase::PolyToStandardBase;

        void runOnOperation() override {
          MLIRContext *context = &getContext();
          auto *module = getOperation();

          ConversionTarget target(*context);
          target.addLegalDialect<arith::ArithDialect>();
          target.addIllegalDialect<PolyDialect>();

          RewritePatternSet patterns(context);
          PolyToStandardTypeConvertor typeConvertor(context);
          patterns.add<ConvertAdd>(typeConvertor, context);

          populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
            patterns, typeConvertor);
          target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConvertor.isSignatureLegal(op.getFunctionType()) &&
	    		typeConvertor.isLegal(&op.getBody());
          });

          populateReturnOpTypeConversionPattern(patterns, typeConvertor);
          target.addDynamicallyLegalOp<func::ReturnOp>(
            [&](func::ReturnOp op) { return typeConvertor.isLegal(op); 
            }
          );
          
          populateCallOpTypeConversionPattern(patterns, typeConvertor);
          target.addDynamicallyLegalOp<func::CallOp>(
              [&](func::CallOp op) { return typeConvertor.isLegal(op); });

          populateBranchOpInterfaceTypeConversionPattern(patterns, typeConvertor);
          target.markUnknownOpDynamicallyLegal([&](Operation *op) {
            return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
                  isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                    typeConvertor) ||
                  isLegalForReturnOpTypeConversionPattern(op, typeConvertor);
          });
           
          if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
          }
        }
      };
    } // namespace poly
  } // namespace heir
} // namespace mlir
