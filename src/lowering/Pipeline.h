//===- Pipeline.h - tc -> LLVM lowering pipelines -----------------*- C++ -*-===//
//
// Stage: lowering. In: the tc IR from import. Out: the LLVM dialect, ready
// for runtime/JIT.h. It is three pass pipelines, run in order:
//
//   tc-lower-to-linalg   tc ops -> Linalg on tensors            (-emit=linalg)
//   tc-bufferize         tensors -> memrefs, Linalg -> loops    (-emit=memref)
//   tc-lower-to-llvm     everything -> LLVM dialect             (-emit=llvm)
//
// Each is also registered with tc-opt under the name above.
//
//===----------------------------------------------------------------------===//

#ifndef TC_LOWERING_PIPELINE_H
#define TC_LOWERING_PIPELINE_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class DialectRegistry;
class OpPassManager;

namespace tc {

enum class PipelineStage { Linalg, Memref, LLVM };

/// Register every dialect (and interface extension) the pipelines touch.
void registerTCCompilerDialects(DialectRegistry &registry);

void buildLowerToLinalg(OpPassManager &pm);
void buildBufferize(OpPassManager &pm);
void buildLowerToLLVM(OpPassManager &pm);

/// Make the three pipelines available to tc-opt by name.
void registerTCPipelines();

/// Run the pipelines on `module` up to and including `stage`.
LogicalResult runPipeline(ModuleOp module, PipelineStage stage);

} // namespace tc
} // namespace mlir

#endif // TC_LOWERING_PIPELINE_H
