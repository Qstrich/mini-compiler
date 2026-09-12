// RUN: not tc-opt %s -split-input-file 2>&1 | FileCheck %s

func.func @add_shape_mismatch(%a: tensor<2x2xf32>, %b: tensor<3x3xf32>) -> tensor<2x2xf32> {
  %y = tc.add %a, %b : tensor<2x2xf32>, tensor<3x3xf32> -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// CHECK: operands must have the same shape or one must be a 0-D scalar

// -----

func.func @add_result_mismatch(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<4x4xf32> {
  %y = tc.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<4x4xf32>
  return %y : tensor<4x4xf32>
}

// CHECK: result type must match the non-scalar operand type

// -----

func.func @matmul_rank(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %m = tc.matmul %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %m : tensor<4xf32>
}

// CHECK: operands must be rank-2 tensors

// -----

func.func @matmul_k_mismatch(%a: tensor<2x3xf32>, %b: tensor<4x5xf32>) -> tensor<2x5xf32> {
  %m = tc.matmul %a, %b : tensor<2x3xf32>, tensor<4x5xf32> -> tensor<2x5xf32>
  return %m : tensor<2x5xf32>
}

// CHECK: contracting dimension mismatch

// -----

func.func @matmul_result_shape(%a: tensor<2x4xf32>, %b: tensor<4x3xf32>) -> tensor<3x2xf32> {
  %m = tc.matmul %a, %b : tensor<2x4xf32>, tensor<4x3xf32> -> tensor<3x2xf32>
  return %m : tensor<3x2xf32>
}

// CHECK: result must have shape 2x3
