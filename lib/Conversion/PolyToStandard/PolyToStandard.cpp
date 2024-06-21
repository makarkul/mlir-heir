#include "lib/Conversion/PolyToStandard/PolyToStandard.h"

#include "lib/Dialect/Poly/PolyOps.h"
#include "lib/Dialect/Poly/PolyTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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

      struct ConvertSub : public OpConversionPattern<PolySubOp> {
        ConvertSub(mlir::MLIRContext *context) 
          : OpConversionPattern<PolySubOp>(context) {}

        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(
          PolySubOp op, OpAdaptor adaptor,
          ConversionPatternRewriter &rewriter) const override {
            arith::SubIOp subOp = rewriter.create<arith::SubIOp>(
              op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
            rewriter.replaceOp(op.getOperation(), subOp);
            return success();
          }
      };

      struct ConvertFromTensor : public OpConversionPattern<PolyFromTensorOp> {
        ConvertFromTensor(mlir::MLIRContext *context)
          : OpConversionPattern<PolyFromTensorOp>(context) {}
        
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(
          PolyFromTensorOp op, OpAdaptor adaptor,
          ConversionPatternRewriter &rewriter) const override {
            auto resultTensorTy = cast<RankedTensorType>(
              typeConverter->convertType(op->getResultTypes()[0])
            );
            auto resultShape = resultTensorTy.getShape()[0];
            auto resultEltTy =resultTensorTy.getElementType();

            auto inputTensorTy = op.getInput().getType();
            auto inputShape = inputTensorTy.getShape()[0];

            // Zero pad the tensor if the coffecients' size is less than the polynomial
            // degree
            ImplicitLocOpBuilder b(op.getLoc(), rewriter);
            auto coeffValue = adaptor.getInput();
            if (inputShape < resultShape) {
              SmallVector<OpFoldResult, 1> low, high;
              low.push_back(rewriter.getIndexAttr(0));
              high.push_back(rewriter.getIndexAttr(resultShape - inputShape));
              coeffValue = b.create<tensor::PadOp>(
                resultTensorTy, coeffValue, low, high,
                b.create<arith::ConstantOp>
                (rewriter.getIntegerAttr(resultEltTy, 0)),
              /*nofold=*/false);
            }
            rewriter.replaceOp(op, coeffValue);
            return success();
          }
      };

      struct ConvertToTensor : public OpConversionPattern<PolyToTensorOp> {
        ConvertToTensor(mlir::MLIRContext *context) 
          : OpConversionPattern<PolyToTensorOp>(context) {}
        
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(
            PolyToTensorOp op, OpAdaptor adaptor,
            ConversionPatternRewriter &rewriter) const override {
          rewriter.replaceOp(op, adaptor.getInput());
          return success();
        }
      };

      struct ConvertConstant : public OpConversionPattern<PolyConstantOp> {
        ConvertConstant(mlir::MLIRContext *context)
          : OpConversionPattern<PolyConstantOp>(context) {}

        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(
            PolyConstantOp op, OpAdaptor adaptor,
            ConversionPatternRewriter &rewriter) const override {
          ImplicitLocOpBuilder b(op.getLoc(), rewriter);
          auto constOp = b.create<arith::ConstantOp>(adaptor.getCoeffecients());
          auto fromTensorOp = 
              b.create<PolyFromTensorOp>(op.getResult().getType(), constOp);
          rewriter.replaceOp(op, fromTensorOp.getResult());
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
          patterns.add<ConvertAdd, ConvertConstant, ConvertSub, ConvertFromTensor, ConvertToTensor>(
            typeConvertor, context);

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
