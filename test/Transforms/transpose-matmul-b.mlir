// RUN: tc-opt %s --tc-transpose-matmul-b | FileCheck %s

func.func @matmul(%a: tensor<2x4xf32>, %b: tensor<4x3xf32>) -> tensor<2x3xf32> {
  %e = tensor.empty() : tensor<2x3xf32>
  %c = linalg.matmul ins(%a, %b : tensor<2x4xf32>, tensor<4x3xf32>)
                     outs(%e : tensor<2x3xf32>) -> tensor<2x3xf32>
  return %c : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @matmul
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [1, 0]
// CHECK: linalg.matmul indexing_maps
// CHECK-SAME: tensor<2x4xf32>, tensor<3x4xf32>
