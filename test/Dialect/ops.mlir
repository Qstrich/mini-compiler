// RUN: tc-opt %s | FileCheck %s

func.func @constants() -> tensor<2x2xf32> {
  %c = tc.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @constants
// CHECK: %[[C:.*]] = tc.constant
// CHECK-SAME: tensor<2x2xf32>
// CHECK: return %[[C]] : tensor<2x2xf32>

func.func @add(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %y = tc.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @add
// CHECK: %[[Y:.*]] = tc.add %{{.*}}, %{{.*}} : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
// CHECK: return %[[Y]] : tensor<2x2xf32>

func.func @add_scalar(%a: tensor<2x2xf32>, %s: tensor<f32>) -> tensor<2x2xf32> {
  %y = tc.add %a, %s : tensor<2x2xf32>, tensor<f32> -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @add_scalar
// CHECK: tc.add %{{.*}}, %{{.*}} : tensor<2x2xf32>, tensor<f32> -> tensor<2x2xf32>

func.func @relu(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %r = tc.relu %x : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @relu
// CHECK: %[[R:.*]] = tc.relu %{{.*}} : tensor<2x2xf32>
// CHECK: return %[[R]] : tensor<2x2xf32>

func.func @matmul(%a: tensor<2x4xf32>, %w: tensor<4x3xf32>) -> tensor<2x3xf32> {
  %m = tc.matmul %a, %w : tensor<2x4xf32>, tensor<4x3xf32> -> tensor<2x3xf32>
  return %m : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @matmul
// CHECK: %[[M:.*]] = tc.matmul %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<4x3xf32> -> tensor<2x3xf32>
// CHECK: return %[[M]] : tensor<2x3xf32>
