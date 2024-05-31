// RUN: heir-opt %s --verify-roundtrip | FileCheck %s

func.func @main(%arg0: i32) {
  %0 = math.ctlz %arg0 : i32
  func.return
}
// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32
// CHECK-SAME:                       ) {
// CHECK:           %[[VAL_1:.*]] = math.ctlz %[[VAL_0]] : i32
// CHECK:           return
// CHECK:         }
