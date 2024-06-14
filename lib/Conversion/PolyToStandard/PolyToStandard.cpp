#include "lib/Conversion/PolyToStandard/PolyToStandard.h"

#include "lib/Dialect/Poly/PolyOps.h"
#include "lib/Dialect/Poly/PolyTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
  namespace heir {
    namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "lib/Conversion/PolyToStandard/PolyToStandard.h.inc"

      struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
        using PolyToStandardBase::PolyToStandardBase;

        void runOnOperation() override {
          MLIRContext *context = &getContext();
          auto *module = getOperation();

          ConversionTarget target(*context);
          target.addIllegalDialect<PolyDialect>();

          RewritePatternSet patterns(context);

          if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
          }
        }
      };
    } // namespace poly
  } // namespace heir
} // namespace mlir
