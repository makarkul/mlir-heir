#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/Passes.h"
#include "lib/Conversion/PolyToStandard/PolyToStandard.h"
#include "lib/Dialect/Poly/PolyDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Poly
  manager.addPass(mlir::heir::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::heir::poly::PolyDialect>();
  mlir::registerAllDialects(registry);

  mlir::registerAllPasses();

  mlir::heir::registerAffinePasses();
  mlir::heir::registerArithPasses();

  // Dialect conversion passes
  mlir::heir::poly::registerPolyToStandardPasses();

  mlir::PassPipelineRegistration<>("poly-to-llvm",
                          "Run passes to lowe the poly dialect to LLVM",
                          polyToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HEIR Pass Driver", registry)
      );
}
