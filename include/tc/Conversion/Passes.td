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

def TransposeMatmulB : Pass<"tc-transpose-matmul-b", "::mlir::ModuleOp"> {
  let summary = "Rewrite linalg.matmul to a transpose-B variant.";
  let description = [{
    Inserts `linalg.transpose` on the RHS and rewrites `linalg.matmul` to
    `matmul_transpose_b` so the contracting K dimension of B is contiguous.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def TileAndFuse : Pass<"tc-tile-and-fuse", "::mlir::ModuleOp"> {
  let summary = "Tile linalg matmul and fuse Add/Relu epilogues.";
  let description = [{
    Tiles `linalg.matmul` (including transpose-B) with the given M,N,K sizes
    using `scf.for`, then fuses elementwise tensor consumers (add / relu)
    into the tile loops. Remaining elementwise generics are tiled on their
    parallel dimensions.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect"
  ];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for M,N,K (0 means do not tile that dim). "
               "Default 32,32,32 when omitted.">
  ];
}

#endif // TC_CONVERSION_PASSES
