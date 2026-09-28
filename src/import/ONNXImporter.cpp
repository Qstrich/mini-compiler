//===- ONNXImporter.cpp - ONNX graph -> ModelInfo -----------------*- C++ -*-===//
//
// Turns a decoded GraphProto into ModelInfo. This is where the ONNX-specific
// restrictions live: static shapes, fp32 only, and `Constant` nodes folded
// into initializers. Op semantics and shapes are checked later by MLIRGen and
// the tc dialect.
//
//===----------------------------------------------------------------------===//

#include "import/ONNXImporter.h"

#include "import/OnnxProto.h"
#include "support/Common.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstring>

using namespace mlir::tc;
using llvm::Error;
using llvm::Expected;
using llvm::StringRef;

/// Graph inputs and outputs must be fp32 with a fully static shape.
static Expected<TensorDesc> importValueInfo(const onnx::ValueInfoProto &info) {
  if (info.name.empty())
    return makeError("graph input/output is missing a name");
  if (info.elemType != onnx::kFloat)
    return makeError("'" + info.name + "' is not FLOAT; only fp32 is supported");
  if (!info.shape)
    return makeError("missing tensor shape for '" + info.name + "'");

  TensorDesc desc;
  desc.name = info.name;
  for (const onnx::Dimension &dim : *info.shape) {
    if (!dim.param.empty())
      return makeError("dynamic shapes are not supported (dim_param='" +
                       dim.param + "')");
    if (!dim.value)
      return makeError("missing static dim_value in shape of '" + info.name +
                       "'");
    if (*dim.value < 0)
      return makeError("negative dimension in shape of '" + info.name + "'");
    desc.shape.push_back(*dim.value);
  }
  return desc;
}

/// Initializers and `Constant` values: fp32 payload in raw_data or float_data.
static Expected<TensorDesc> importTensor(const onnx::TensorProto &tensor,
                                         StringRef name) {
  if (tensor.dataType != onnx::kFloat)
    return makeError("tensor '" + name + "' is not FLOAT; only fp32 is supported");

  TensorDesc desc;
  desc.name = name.str();
  desc.shape = tensor.dims;
  size_t count = static_cast<size_t>(numElements(desc.shape));

  if (!tensor.rawData.empty()) {
    if (tensor.rawData.size() != count * sizeof(float))
      return makeError("raw_data size does not match shape for '" + name + "'");
    desc.data.resize(count);
    std::memcpy(desc.data.data(), tensor.rawData.data(), tensor.rawData.size());
  } else if (tensor.floatData.size() == count) {
    desc.data = tensor.floatData;
  } else {
    return makeError("float_data count does not match shape for '" + name +
                     "'");
  }
  return desc;
}

/// `Constant` carries its tensor in the `value` attribute.
static Expected<TensorDesc> importConstantNode(const onnx::NodeProto &node) {
  if (node.outputs.size() != 1 || node.outputs[0].empty())
    return makeError("Constant node must have one named output");
  for (const onnx::AttributeProto &attr : node.attributes)
    if (attr.name == "value" && attr.t)
      return importTensor(*attr.t, node.outputs[0]);
  return makeError("Constant node '" + node.outputs[0] +
                   "' is missing a tensor 'value' attribute");
}

static Expected<ModelInfo> importGraph(const onnx::GraphProto &graph) {
  ModelInfo model;
  model.name = graph.name;

  llvm::StringSet<> constantNames;
  auto addConstant = [&](Expected<TensorDesc> desc) -> Error {
    if (!desc)
      return desc.takeError();
    if (!constantNames.insert(desc->name).second)
      return makeError("duplicate initializer '" + desc->name + "'");
    model.initializers.push_back(std::move(*desc));
    return Error::success();
  };

  for (const onnx::TensorProto &init : graph.initializers)
    if (Error err = addConstant(importTensor(init, init.name)))
      return std::move(err);

  // Graph inputs that are also initializers are defaults, not arguments.
  for (const onnx::ValueInfoProto &input : graph.inputs) {
    if (constantNames.contains(input.name))
      continue;
    Expected<TensorDesc> desc = importValueInfo(input);
    if (!desc)
      return desc.takeError();
    model.inputs.push_back(std::move(*desc));
  }

  for (const onnx::ValueInfoProto &output : graph.outputs) {
    Expected<TensorDesc> desc = importValueInfo(output);
    if (!desc)
      return desc.takeError();
    model.outputs.push_back(std::move(*desc));
  }
  if (model.outputs.empty())
    return makeError("graph has no outputs");

  for (const onnx::NodeProto &node : graph.nodes) {
    if (node.opType == "Constant") {
      if (Error err = addConstant(importConstantNode(node)))
        return std::move(err);
      continue;
    }
    model.nodes.push_back({node.opType, node.inputs, node.outputs});
  }
  return model;
}

Expected<ModelInfo> mlir::tc::parseONNXFile(StringRef path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return makeError("failed to open '" + path +
                     "': " + file.getError().message());

  Expected<onnx::GraphProto> graph = onnx::decodeModel((*file)->getBuffer());
  if (!graph)
    return graph.takeError();
  return importGraph(*graph);
}
