//===- TCOps.td - TC dialect ops ---------------------------*- tablegen -*-===//
//
// High-level tensor ops: constant, add, relu, matmul (static fp32).
//
//===----------------------------------------------------------------------===//

#ifndef TC_OPS
#define TC_OPS

include "tc/Dialect/TCDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TC_ConstantOp : TC_Op<"constant", [Pure, ConstantLike,
                                       AllTypesMatch<["value", "result"]>]> {
  let summary = "Dense fp32 tensor constant.";
  let description = [{
    Materializes a static `tensor<...xf32>` from a dense elements attribute.

    ```mlir
    %c = tc.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    ```
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs TC_F32Tensor:$result);
  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def TC_AddOp : TC_Op<"add", [Pure]> {
  let summary = "Elementwise add with same-shape or 0-D scalar broadcast.";
  let description = [{
    Adds two ranked `f32` tensors. Shapes must match, or one operand must be
    a 0-D `tensor<f32>` (scalar broadcast). The result type is the non-scalar
    operand type (or the shared type if both have the same shape).

    ```mlir
    %y = tc.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    %z = tc.add %a, %s : tensor<2x2xf32>, tensor<f32> -> tensor<2x2xf32>
    ```
  }];
  let arguments = (ins TC_F32Tensor:$lhs, TC_F32Tensor:$rhs);
  let results = (outs TC_F32Tensor:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def TC_ReluOp : TC_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise ReLU: max(x, 0).";
  let description = [{
    Applies ReLU to a ranked `f32` tensor. Input and result types must match.

    ```mlir
    %r = tc.relu %y : tensor<2x2xf32>
    ```
  }];
  let arguments = (ins TC_F32Tensor:$input);
  let results = (outs TC_F32Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def TC_MatMulOp : TC_Op<"matmul", [Pure]> {
  let summary = "Rank-2 matrix multiply: A[M,K] @ B[K,N] -> C[M,N].";
  let description = [{
    Multiplies two rank-2 `f32` tensors. The contracting dimension `K` must
    match; the result is `tensor<MxNxf32>`.

    ```mlir
    %m = tc.matmul %a, %w : tensor<2x4xf32>, tensor<4x3xf32> -> tensor<2x3xf32>
    ```
  }];
  let arguments = (ins TC_F32Tensor:$lhs, TC_F32Tensor:$rhs);
  let results = (outs TC_F32Tensor:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

#endif // TC_OPS
