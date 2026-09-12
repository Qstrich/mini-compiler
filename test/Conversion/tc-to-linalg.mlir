// RUN: tc-opt %s --convert-tc-to-linalg | FileCheck %s

func.func @constants() -> tensor<2x2xf32> {
  %c = tc.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @constants
// CHECK: arith.constant dense<{{.*}}> : tensor<2x2xf32>
// CHECK-NOT: tc.constant

func.func @add(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %y = tc.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @add
// CHECK: tensor.empty() : tensor<2x2xf32>
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK-NOT: tc.add

func.func @add_scalar(%a: tensor<2x2xf32>, %s: tensor<f32>) -> tensor<2x2xf32> {
  %y = tc.add %a, %s : tensor<2x2xf32>, tensor<f32> -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @add_scalar
// CHECK: linalg.generic
// CHECK: arith.addf

func.func @relu(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %r = tc.relu %x : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @relu
// CHECK: linalg.generic
// CHECK: arith.maximumf
// CHECK-NOT: tc.relu

func.func @matmul(%a: tensor<2x4xf32>, %w: tensor<4x3xf32>) -> tensor<2x3xf32> {
  %m = tc.matmul %a, %w : tensor<2x4xf32>, tensor<4x3xf32> -> tensor<2x3xf32>
  return %m : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @matmul
// CHECK: tensor.empty() : tensor<2x3xf32>
// CHECK: linalg.matmul
// CHECK-NOT: tc.matmul
