#ifndef LIB_TRANSFORM_ARITH_MULTTOADD_H_
#define LIB_TRANSFORM_ARITH_MULTTOADD_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace heir {

    class MulToAddPass
      : public PassWrapper<MulToAddPass,
                            OperationPass<mlir::func::FuncOp>> {

    private:
      void runOnOperation() override;

      StringRef getArgument() const final { return "mul-to-add"; }

      StringRef getDescription() const final {
        return "Convert multiplication to repeated additions";
      }
    };
  } // namespace heir
} // namespace mlir


#endif//LIB_TRANSFORM_ARITH_MULTTOADD_H_
