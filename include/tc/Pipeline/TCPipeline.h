//===- TCPipeline.h - Progressive lowering pipeline --------------*- C++ -*-===//

#ifndef TC_PIPELINE_TCPIPELINE_H
#define TC_PIPELINE_TCPIPELINE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace mlir {
class DialectRegistry;

namespace tc {

enum class PipelineStage {
  Linalg,
  Memref,
  LLVM,
  NVPTX,
};

/// Register dialects needed to parse and lower `tc` IR through LLVM.
void registerTCCompilerDialects(DialectRegistry &registry);

/// Lower `module` through `stage` (inclusive). Returns failure on pass errors.
/// `tileSizes` are M,N,K for Linalg opts (empty means {32, 32, 32}).
LogicalResult runPipeline(ModuleOp module, PipelineStage stage,
                          ArrayRef<int64_t> tileSizes = {});

struct TensorBuffer {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

/// JIT-compile the LLVM-lowered module and invoke `@main`.
/// `inputs` are graph inputs; `outputs` must already have the result shapes
/// (data is overwritten). Argument order is inputs then output out-params.
llvm::Error runJIT(ModuleOp module, ArrayRef<TensorBuffer> inputs,
                   MutableArrayRef<TensorBuffer> outputs);

/// Print `buffer` as a shape header plus row-major floats.
void printTensorBuffer(const TensorBuffer &buffer, raw_ostream &os);

/// Fill `shape` with 0, 1, 2, ... in row-major order.
TensorBuffer makeSequentialInput(ArrayRef<int64_t> shape);

/// Load a C-contiguous float32 `.npy` file.
llvm::Expected<TensorBuffer> loadNpyF32(StringRef path);

/// After `-emit=nvptx` lowering, write serialized PTX from `gpu.binary` ops.
LogicalResult emitPTX(ModuleOp module, raw_ostream &os);

} // namespace tc
} // namespace mlir

#endif // TC_PIPELINE_TCPIPELINE_H
