//===- ModelInfo.cpp - ModelInfo dump -----------------------------*- C++ -*-===//

#include "tc/Frontend/ModelInfo.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir::tc;

std::string mlir::tc::formatTensorType(llvm::ArrayRef<int64_t> shape) {
  std::string out = "tensor<";
  if (shape.empty()) {
    out += "f32";
  } else {
    for (auto [i, dim] : llvm::enumerate(shape)) {
      if (i)
        out += "x";
      out += std::to_string(dim);
    }
    out += "xf32";
  }
  out += ">";
  return out;
}

void mlir::tc::dumpModelInfo(const ModelInfo &model, llvm::raw_ostream &os) {
  os << "model: " << (model.name.empty() ? "<unnamed>" : model.name) << "\n";

  os << "inputs:\n";
  if (model.inputs.empty())
    os << "  (none)\n";
  for (const TensorDesc &t : model.inputs)
    os << "  " << t.name << ": " << formatTensorType(t.shape) << "\n";

  os << "initializers:\n";
  if (model.initializers.empty())
    os << "  (none)\n";
  for (const TensorDesc &t : model.initializers)
    os << "  " << t.name << ": " << formatTensorType(t.shape) << "\n";

  os << "outputs:\n";
  if (model.outputs.empty())
    os << "  (none)\n";
  for (const TensorDesc &t : model.outputs)
    os << "  " << t.name << ": " << formatTensorType(t.shape) << "\n";

  os << "nodes:\n";
  if (model.nodes.empty())
    os << "  (none)\n";
  for (auto [i, node] : llvm::enumerate(model.nodes)) {
    os << "  " << i << ": " << node.opType << "(";
    llvm::interleaveComma(node.inputs, os);
    os << ") -> ";
    llvm::interleaveComma(node.outputs, os);
    os << "\n";
  }
}
