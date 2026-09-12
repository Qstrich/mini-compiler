//===- ModelInfo.h - IR-agnostic ONNX graph -----------------------*- C++ -*-===//

#ifndef TC_FRONTEND_MODELINFO_H
#define TC_FRONTEND_MODELINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace tc {

struct TensorDesc {
  std::string name;
  std::vector<int64_t> shape;
  /// Populated for initializers and Constant nodes; empty for graph inputs.
  std::vector<float> data;
};

struct NodeInfo {
  std::string opType;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

struct ModelInfo {
  std::string name;
  /// Graph inputs that are not also initializers (function arguments).
  std::vector<TensorDesc> inputs;
  std::vector<TensorDesc> outputs;
  std::vector<TensorDesc> initializers;
  std::vector<NodeInfo> nodes;
};

/// Print `tensor<2x4xf32>` / `tensor<f32>` for FileCheck-friendly dumps.
std::string formatTensorType(llvm::ArrayRef<int64_t> shape);

void dumpModelInfo(const ModelInfo &model, llvm::raw_ostream &os);

} // namespace tc
} // namespace mlir

#endif // TC_FRONTEND_MODELINFO_H
