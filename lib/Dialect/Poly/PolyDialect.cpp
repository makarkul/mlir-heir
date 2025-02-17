#include "lib/Dialect/Poly/PolyDialect.h"

#include "lib/Dialect/Poly/PolyOps.h"
#include "lib/Dialect/Poly/PolyTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/Poly/PolyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poly/PolyTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Poly/PolyOps.cpp.inc"

namespace mlir {
  namespace heir {
    namespace poly {

      void PolyDialect::initialize() {
        // This is where we will register types and operations with the dialect
        addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Poly/PolyTypes.cpp.inc"
          >();
          
        addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Poly/PolyOps.cpp.inc"
          >();
      } 

      Operation *PolyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                          Type type, Location loc) {
        auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
        if (!coeffs) {
          return nullptr;
        }
        return builder.create<PolyConstantOp>(loc, type, coeffs);
      }
    }// namespace poly 
  }// namespace heir
}// namespace mlir