//===- MLIRGen.h - Lower ModelInfo to tc dialect ------------------*- C++ -*-===//

#ifndef TC_FRONTEND_MLIRGEN_H
#define TC_FRONTEND_MLIRGEN_H

#include "tc/Frontend/ModelInfo.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/Error.h"

namespace mlir {

class MLIRContext;

namespace tc {

/// Build `func.func @main` with tc dialect ops from a parsed ModelInfo.
llvm::Expected<OwningOpRef<ModuleOp>> generateMLIR(MLIRContext &context,
                                                   const ModelInfo &model,
                                                   llvm::StringRef filename);

} // namespace tc
} // namespace mlir

#endif // TC_FRONTEND_MLIRGEN_H
