#include "lib/Conversion/PolyToStandard/PolyToStandard.h"

#include "lib/Dialect/Poly/PolyOps.h"
#include "lib/Dialect/Poly/PolyTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

          // Convert from a tensor type to a poly type: use from_tensor
          addSourceMaterialization([](OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) -> Value {
            return builder.create<poly::PolyFromTensorOp>(loc, type, inputs[0]);
          });
          // Convert from a tensor type to a poly type: use to_tensor
          addSourceMaterialization([](OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) -> Value {
            return builder.create<poly::PolyToTensorOp>(loc, type, inputs[0]);
          });
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

      struct ConvertMul : public OpConversionPattern<PolyMulOp> {
        ConvertMul(mlir::MLIRContext *context)
            : OpConversionPattern<PolyMulOp>(context) {}

        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(
              PolyMulOp op, OpAdaptor adaptor,
              ConversionPatternRewriter &rewriter) const override {
          auto polymulTensorType = cast<RankedTensorType>(adaptor.getLhs().getType());
          auto numTerms = polymulTensorType.getShape()[0];
          ImplicitLocOpBuilder b(op.getLoc(), rewriter);

          // Create an all-zeros tensor to store the result
          auto polymulResult = b.create<arith::ConstantOp>(
            polymulTensorType, DenseElementsAttr::get(polymulTensorType, 0)
          );

          // Loop bounds and step
          auto lowerBound =
              b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(0));
          auto numTermsOp =
              b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(numTerms));
          auto step =
              b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));

          auto p0 = adaptor.getLhs();
          auto p1 = adaptor.getRhs();

          // for i = 0, ..., N-1
          //   for j = 0, ..., N-1
          //      product[i+j (mod N)] += p0[i] * p1[j]
          auto outerloop = b.create<scf::ForOp>(
            lowerBound, numTermsOp, step, ValueRange(polymulResult.getResult()),
            [&](OpBuilder &builder, Location loc, Value p0Index,
                ValueRange loopState) {
                  ImplicitLocOpBuilder b(op.getLoc(), builder);
                  
              auto innerLoop = b.create<scf::ForOp>(
                lowerBound, numTermsOp, step, loopState,
                [&](OpBuilder &builder, Location loc, Value p1Index,
                    ValueRange loopState) {
                  ImplicitLocOpBuilder b(op.getLoc(), builder);
                  auto accumTensor = loopState.front();
                  auto destIndex = b.create<arith::RemUIOp>(
                    b.create<arith::AddIOp>(p0Index, p1Index), numTermsOp
                  );
                  auto mulOp = b.create<arith::MulIOp>(
                    b.create<tensor::ExtractOp>(p0, ValueRange(p0Index)),
                    b.create<tensor::ExtractOp>(p1, ValueRange(p1Index))
                  );
                  auto result = b.create<arith::AddIOp>(
                    mulOp, b.create<tensor::ExtractOp>(accumTensor,
                                                        destIndex.getResult())
                  );
                  auto stored = b.create<tensor::InsertOp>(result, accumTensor,
                                                            destIndex.getResult()
                  );
                  b.create<scf::YieldOp>(stored.getResult());
                }
              );
              b.create<scf::YieldOp>(innerLoop.getResults());
            }
          );
          rewriter.replaceOp(op, outerloop.getResult(0));
          return success();
        }
      };

      struct ConvertEval : public OpConversionPattern<PolyEvalOp> {
        ConvertEval(mlir::MLIRContext *context)
            : OpConversionPattern<PolyEvalOp>(context) {}

        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(
            PolyEvalOp op, OpAdaptor adaptor,
            ConversionPatternRewriter &rewriter) const override {
          auto polyTensorType =
              cast<RankedTensorType>(adaptor.getPolynomial().getType());
          auto numTerms = polyTensorType.getShape()[0];
          ImplicitLocOpBuilder b(op.getLoc(), rewriter);

          auto lowerBound =
              b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));
          auto numTermsOp = b.create<arith::ConstantOp>(b.getIndexType(),
                                                        b.getIndexAttr(numTerms + 1));
          auto step = lowerBound;

          auto poly = adaptor.getPolynomial();
          auto point = adaptor.getPoint();

          // Horner's method:
          // 
          // accum = 0
          // for i = 1, 2, ..., N
          //    accum = point * accum + coeff[N - i]
          auto accum =
            b.create<arith::ConstantOp>(b.getI32Type(), b.getI32IntegerAttr(0));
          auto loop = b.create<scf::ForOp>(
              lowerBound, numTermsOp, step, accum.getResult(),
              [&](OpBuilder &builder, Location loc, Value loopIndex,
                ValueRange loopState) {

              ImplicitLocOpBuilder b(op.getLoc(), builder);
              auto accum = loopState.front();
              auto coeffIndex = b.create<arith::SubIOp>(numTermsOp, loopIndex);
              auto mulOp = b.create<arith::MulIOp>(point, accum);
              auto result = b.create<arith::AddIOp>(
                  mulOp, b.create<tensor::ExtractOp>(poly, coeffIndex.getResult()));
              b.create<scf::YieldOp>(result.getResult());
            });

          rewriter.replaceOp(op, loop.getResult(0));
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
          target.addIllegalOp<PolyAddOp, PolyMulOp, PolyEvalOp, PolyConstantOp, 
            PolyFromTensorOp, PolyToTensorOp>();

          RewritePatternSet patterns(context);
          PolyToStandardTypeConvertor typeConvertor(context);
          patterns.add<ConvertAdd, ConvertConstant, ConvertEval, ConvertMul, 
             ConvertFromTensor, ConvertToTensor>(typeConvertor, context);

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
