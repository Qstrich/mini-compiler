// RUN: tc-opt %s --tc-outline-gpu-kernel | FileCheck %s

func.func @main(%a: memref<2x2xf32>, %b: memref<2x2xf32>) {
  %c0 = arith.constant 0 : index
  %v = memref.load %a[%c0, %c0] : memref<2x2xf32>
  memref.store %v, %b[%c0, %c0] : memref<2x2xf32>
  return
}

// CHECK-LABEL: func.func @main
// CHECK: gpu.launch
// CHECK: memref.load
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK: return
