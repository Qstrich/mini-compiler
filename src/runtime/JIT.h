//===- JIT.h - Run a lowered module on the host CPU ---------------*- C++ -*-===//
//
// Stage: runtime.
//   In:   a module lowered to the LLVM dialect, plus host input tensors.
//   Out:  the output tensors, computed by native code on this CPU.
//
//===----------------------------------------------------------------------===//

#ifndef TC_RUNTIME_JIT_H
#define TC_RUNTIME_JIT_H

#include "runtime/TensorBuffer.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace tc {

/// JIT-compile `module` (already lowered to the LLVM dialect) and call `@main`
/// with `inputs`. Allocates one output per entry of `outputShapes`, passes
/// them as @main's trailing out-params, and returns them filled.
llvm::Expected<llvm::SmallVector<TensorBuffer>>
runJIT(ModuleOp module, llvm::ArrayRef<TensorBuffer> inputs,
       llvm::ArrayRef<std::vector<int64_t>> outputShapes);

} // namespace tc
} // namespace mlir

#endif // TC_RUNTIME_JIT_H
