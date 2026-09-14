// RUN: tc-opt %s --gpu-map-parallel-loops --convert-parallel-loops-to-gpu | FileCheck %s

func.func @map(%a: memref<2x3xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c3) step (%c1, %c1) {
    %v = memref.load %a[%i, %j] : memref<2x3xf32>
    memref.store %v, %a[%i, %j] : memref<2x3xf32>
  }
  return
}

// CHECK-LABEL: func.func @map
// CHECK: gpu.launch blocks
// CHECK-SAME: threads
// CHECK: memref.load
// CHECK: gpu.terminator
