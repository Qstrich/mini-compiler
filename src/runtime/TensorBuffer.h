//===- TensorBuffer.h - Host-side fp32 tensors --------------------*- C++ -*-===//

#ifndef TC_RUNTIME_TENSORBUFFER_H
#define TC_RUNTIME_TENSORBUFFER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <vector>

namespace mlir {
namespace tc {

/// A row-major fp32 tensor owned by the host.
struct TensorBuffer {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

/// A tensor of `shape` filled with 0, 1, 2, ... in row-major order.
TensorBuffer makeSequentialTensor(llvm::ArrayRef<int64_t> shape);

/// Load a C-contiguous little-endian float32 `.npy` file.
llvm::Expected<TensorBuffer> loadNpyF32(llvm::StringRef path);

/// Print a `shape:2x3` header, then one line of floats per innermost row.
void printTensorBuffer(const TensorBuffer &buffer, llvm::raw_ostream &os);

} // namespace tc
} // namespace mlir

#endif // TC_RUNTIME_TENSORBUFFER_H
