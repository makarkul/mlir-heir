#include "lib/Transform/Affine/AffineFullUnroll.h"
#include "lib/Transform/Arith/MulToAdd.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"


int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::registerAllPasses();

  mlir::PassRegistration<mlir::heir::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::heir::AffineFullUnrollPassAsPatternRewrite>();
  mlir::PassRegistration<mlir::heir::MulToAddPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HEIR Pass Driver", registry)
      );
}
