// RUN: tc-compile %tc_src/models/add.onnx -emit=mlir | FileCheck %s --check-prefix=ADD
// RUN: tc-compile %tc_src/models/relu.onnx -emit=mlir | FileCheck %s --check-prefix=RELU
// RUN: tc-compile %tc_src/models/matmul.onnx -emit=mlir | FileCheck %s --check-prefix=MATMUL
// RUN: tc-compile %tc_src/models/linear.onnx -emit=mlir | FileCheck %s --check-prefix=LINEAR
// RUN: tc-compile %tc_src/models/linear.onnx -emit=proto | FileCheck %s --check-prefix=PROTO

// ADD: func.func @main(%{{.*}}: tensor<2x2xf32>, %{{.*}}: tensor<2x2xf32>) -> tensor<2x2xf32>
// ADD: tc.add %{{.*}}, %{{.*}} : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
// ADD: return

// RELU: func.func @main(%{{.*}}: tensor<2x2xf32>) -> tensor<2x2xf32>
// RELU: tc.relu %{{.*}} : tensor<2x2xf32>
// RELU: return

// MATMUL: func.func @main(%{{.*}}: tensor<2x4xf32>, %{{.*}}: tensor<4x3xf32>) -> tensor<2x3xf32>
// MATMUL: tc.matmul %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<4x3xf32> -> tensor<2x3xf32>
// MATMUL: return

// LINEAR: func.func @main(%{{.*}}: tensor<2x4xf32>) -> tensor<2x3xf32>
// LINEAR: tc.constant
// LINEAR: tc.constant
// LINEAR: tc.matmul
// LINEAR: tc.add
// LINEAR: tc.relu
// LINEAR: return

// PROTO: model: linear
// PROTO: inputs:
// PROTO: X: tensor<2x4xf32>
// PROTO: initializers:
// PROTO: W: tensor<4x3xf32>
// PROTO: b: tensor<2x3xf32>
// PROTO: outputs:
// PROTO: Y: tensor<2x3xf32>
// PROTO: nodes:
// PROTO: MatMul(X, W) -> t1
// PROTO: Add(t1, b) -> t2
// PROTO: Relu(t2) -> Y
