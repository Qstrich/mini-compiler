// RUN: tc-opt %s --gpu-map-parallel-loops | FileCheck %s --check-prefix=MAP
// RUN: tc-opt %s --gpu-map-parallel-loops --convert-parallel-loops-to-gpu | FileCheck %s --check-prefix=LAUNCH

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

// LAUNCH-LABEL: func.func @map
// LAUNCH: gpu.launch blocks
// LAUNCH-SAME: threads
// LAUNCH: memref.load
// LAUNCH: gpu.terminator

func.func @nested_map(%a: memref<2x3xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  scf.parallel (%bi, %bj) = (%c0, %c0) to (%c2, %c3) step (%c1, %c1) {
    scf.parallel (%ti, %tj) = (%c0, %c0) to (%c2, %c3) step (%c1, %c1) {
      %v = memref.load %a[%ti, %tj] : memref<2x3xf32>
      memref.store %v, %a[%ti, %tj] : memref<2x3xf32>
    }
  }
  return
}

// MAP-LABEL: func.func @nested_map
// MAP: scf.parallel
// MAP: scf.parallel
// MAP: mapping = [#gpu.loop_dim_map<processor = thread_
// MAP: mapping = [#gpu.loop_dim_map<processor = block_

// LAUNCH-LABEL: func.func @nested_map
// LAUNCH: gpu.launch blocks
// LAUNCH-SAME: threads
// LAUNCH: memref.load
// LAUNCH: gpu.terminator
