//===- Passes.td - TC conversion passes ----------------------*- tablegen -*-===//

#ifndef TC_CONVERSION_PASSES
#define TC_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTCToLinalg : Pass<"convert-tc-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Convert tc dialect ops to linalg / arith on tensors.";
  let description = [{
    Lowers `tc.constant`, `tc.add`, `tc.relu`, and `tc.matmul` to
    destination-passing `linalg` / `arith` ops on ranked `f32` tensors.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // TC_CONVERSION_PASSES
