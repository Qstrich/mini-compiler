//===- MLIRGen.h - Lower ModelInfo to tc dialect ------------------*- C++ -*-===//
//
// Stage: import, step 2 of 2 (named after the MLIR Toy tutorial's MLIRGen).
//   In:   ModelInfo from parseONNXFile.
//   Out:  a module holding `func.func @main` built from tc ops.
//         Next: runPipeline (lowering/Pipeline.h).
//
//===----------------------------------------------------------------------===//

#ifndef TC_IMPORT_MLIRGEN_H
#define TC_IMPORT_MLIRGEN_H

#include "import/ModelInfo.h"

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

#endif // TC_IMPORT_MLIRGEN_H
