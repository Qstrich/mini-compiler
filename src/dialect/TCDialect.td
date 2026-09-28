//===- TCDialect.td - TC dialect ---------------------------*- tablegen -*-===//
//
// High-level tensor dialect for the mini tensor compiler.
// Holds ONNX-derived tensor ops before lowering to linalg / memref / LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef TC_DIALECT
#define TC_DIALECT

include "mlir/IR/OpBase.td"

def TC_Dialect : Dialect {
  let name = "tc";
  let summary = "High-level tensor dialect for the mini compiler.";
  let description = [{
    Frontend dialect that holds ONNX-derived tensor ops before lowering
    to linalg / memref / LLVM.
  }];
  let cppNamespace = "::mlir::tc";
}

class TC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TC_Dialect, mnemonic, traits>;

def TC_F32Tensor : RankedTensorOf<[F32]>;

#endif // TC_DIALECT
