//===- ONNXParser.cpp - ONNX protobuf -> ModelInfo ----------------*- C++ -*-===//

#include "tc/Frontend/ONNXParser.h"
#include "ProtobufReader.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstring>
#include <utility>

using namespace mlir::tc;
using namespace mlir::tc::proto;

// Field numbers match third_party/onnx/onnx.proto (ONNX v1.17.0).
enum {
  AttrName = 1,
  AttrTensor = 5,
  AttrType = 20,

  ValueName = 1,
  ValueType = 2,

  NodeInput = 1,
  NodeOutput = 2,
  NodeName = 3,
  NodeOpType = 4,
  NodeAttribute = 5,

  ModelGraph = 7,

  GraphNode = 1,
  GraphName = 2,
  GraphInitializer = 5,
  GraphInput = 11,
  GraphOutput = 12,
  GraphValueInfo = 13,

  TensorDims = 1,
  TensorDataType = 2,
  TensorFloatData = 4,
  TensorName = 8,
  TensorRawData = 9,

  ShapeDim = 1,
  DimValue = 1,
  DimParam = 2,

  TypeTensor = 1,
  TensorElemType = 1,
  TensorShape = 2,

  OnnxFloat = 1,
};

static llvm::Error makeError(const llvm::Twine &msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
}

struct RawTensor {
  std::string name;
  std::vector<int64_t> dims;
  int32_t dataType = 0;
  std::vector<float> floatData;
  std::string rawData;
};

struct RawAttribute {
  std::string name;
  RawTensor tensor;
  bool hasTensor = false;
};

struct RawNode {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::string name;
  std::string opType;
  std::vector<RawAttribute> attributes;
};

struct RawValueInfo {
  std::string name;
  std::vector<int64_t> shape;
  int32_t elemType = 0;
  bool hasShape = false;
};

struct RawGraph {
  std::string name;
  std::vector<RawNode> nodes;
  std::vector<RawTensor> initializers;
  std::vector<RawValueInfo> inputs;
  std::vector<RawValueInfo> outputs;
  std::vector<RawValueInfo> valueInfos;
};

static llvm::Error expectWire(WireType got, WireType want, const char *what) {
  if (got != want)
    return makeError(llvm::Twine("unexpected wire type for ") + what);
  return llvm::Error::success();
}

static llvm::Error parseDimension(llvm::StringRef bytes, int64_t &dimValue,
                                  std::string &dimParam, bool &hasValue) {
  Reader in(bytes);
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == DimValue) {
      if (auto err = expectWire(wt, WireType::Varint, "dim_value"))
        return err;
      auto v = in.readVarint();
      if (!v)
        return v.takeError();
      dimValue = static_cast<int64_t>(*v);
      hasValue = true;
    } else if (field == DimParam) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "dim_param"))
        return err;
      auto s = in.readBytes();
      if (!s)
        return s.takeError();
      dimParam = s->str();
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  return llvm::Error::success();
}

static llvm::Expected<std::vector<int64_t>>
parseShape(llvm::StringRef bytes) {
  Reader in(bytes);
  std::vector<int64_t> shape;
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field != ShapeDim) {
      if (auto err = in.skip(wt))
        return err;
      continue;
    }
    if (auto err = expectWire(wt, WireType::LengthDelimited, "shape.dim"))
      return err;
    auto dimBytes = in.readBytes();
    if (!dimBytes)
      return dimBytes.takeError();
    int64_t dimValue = 0;
    std::string dimParam;
    bool hasValue = false;
    if (auto err = parseDimension(*dimBytes, dimValue, dimParam, hasValue))
      return err;
    if (!dimParam.empty())
      return makeError("dynamic shapes are not supported (dim_param='" +
                       dimParam + "')");
    if (!hasValue)
      return makeError("missing static dim_value in tensor shape");
    if (dimValue < 0)
      return makeError("negative dimension is not supported");
    shape.push_back(dimValue);
  }
  return shape;
}

static llvm::Error parseTensorType(llvm::StringRef bytes, RawValueInfo &info) {
  Reader in(bytes);
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == TensorElemType) {
      if (auto err = expectWire(wt, WireType::Varint, "elem_type"))
        return err;
      auto v = in.readVarint();
      if (!v)
        return v.takeError();
      info.elemType = static_cast<int32_t>(*v);
    } else if (field == TensorShape) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "shape"))
        return err;
      auto shapeBytes = in.readBytes();
      if (!shapeBytes)
        return shapeBytes.takeError();
      auto shapeOr = parseShape(*shapeBytes);
      if (!shapeOr)
        return shapeOr.takeError();
      info.shape = std::move(*shapeOr);
      info.hasShape = true;
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  return llvm::Error::success();
}

static llvm::Error parseType(llvm::StringRef bytes, RawValueInfo &info) {
  Reader in(bytes);
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == TypeTensor) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "tensor_type"))
        return err;
      auto inner = in.readBytes();
      if (!inner)
        return inner.takeError();
      if (auto err = parseTensorType(*inner, info))
        return err;
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  return llvm::Error::success();
}

static llvm::Error parseValueInfo(llvm::StringRef bytes, RawValueInfo &info) {
  Reader in(bytes);
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == ValueName) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "value.name"))
        return err;
      auto s = in.readBytes();
      if (!s)
        return s.takeError();
      info.name = s->str();
    } else if (field == ValueType) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "value.type"))
        return err;
      auto inner = in.readBytes();
      if (!inner)
        return inner.takeError();
      if (auto err = parseType(*inner, info))
        return err;
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  return llvm::Error::success();
}

static llvm::Error appendPackedFloats(llvm::StringRef bytes,
                                      std::vector<float> &out) {
  if (bytes.size() % 4 != 0)
    return makeError("packed float_data size is not a multiple of 4");
  size_t n = bytes.size() / 4;
  size_t old = out.size();
  out.resize(old + n);
  std::memcpy(out.data() + old, bytes.data(), bytes.size());
  return llvm::Error::success();
}

static llvm::Error parseTensor(llvm::StringRef bytes, RawTensor &tensor) {
  Reader in(bytes);
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == TensorDims) {
      if (auto err = expectWire(wt, WireType::Varint, "dims"))
        return err;
      auto v = in.readVarint();
      if (!v)
        return v.takeError();
      tensor.dims.push_back(static_cast<int64_t>(*v));
    } else if (field == TensorDataType) {
      if (auto err = expectWire(wt, WireType::Varint, "data_type"))
        return err;
      auto v = in.readVarint();
      if (!v)
        return v.takeError();
      tensor.dataType = static_cast<int32_t>(*v);
    } else if (field == TensorFloatData) {
      // packed=true uses length-delimited; unpacked would be fixed32 repeats.
      if (wt == WireType::LengthDelimited) {
        auto packed = in.readBytes();
        if (!packed)
          return packed.takeError();
        if (auto err = appendPackedFloats(*packed, tensor.floatData))
          return err;
      } else if (wt == WireType::Fixed32) {
        auto bits = in.readFixed32();
        if (!bits)
          return bits.takeError();
        float value;
        std::memcpy(&value, &*bits, 4);
        tensor.floatData.push_back(value);
      } else {
        return makeError("unexpected wire type for float_data");
      }
    } else if (field == TensorName) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "tensor.name"))
        return err;
      auto s = in.readBytes();
      if (!s)
        return s.takeError();
      tensor.name = s->str();
    } else if (field == TensorRawData) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "raw_data"))
        return err;
      auto s = in.readBytes();
      if (!s)
        return s.takeError();
      tensor.rawData = s->str();
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  return llvm::Error::success();
}

static llvm::Error parseAttribute(llvm::StringRef bytes, RawAttribute &attr) {
  Reader in(bytes);
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == AttrName) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "attr.name"))
        return err;
      auto s = in.readBytes();
      if (!s)
        return s.takeError();
      attr.name = s->str();
    } else if (field == AttrTensor) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "attr.t"))
        return err;
      auto inner = in.readBytes();
      if (!inner)
        return inner.takeError();
      if (auto err = parseTensor(*inner, attr.tensor))
        return err;
      attr.hasTensor = true;
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  return llvm::Error::success();
}

static llvm::Error parseNode(llvm::StringRef bytes, RawNode &node) {
  Reader in(bytes);
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == NodeInput || field == NodeOutput || field == NodeName ||
        field == NodeOpType) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "node string"))
        return err;
      auto s = in.readBytes();
      if (!s)
        return s.takeError();
      if (field == NodeInput)
        node.inputs.push_back(s->str());
      else if (field == NodeOutput)
        node.outputs.push_back(s->str());
      else if (field == NodeName)
        node.name = s->str();
      else
        node.opType = s->str();
    } else if (field == NodeAttribute) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "attribute"))
        return err;
      auto inner = in.readBytes();
      if (!inner)
        return inner.takeError();
      RawAttribute attr;
      if (auto err = parseAttribute(*inner, attr))
        return err;
      node.attributes.push_back(std::move(attr));
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  return llvm::Error::success();
}

static llvm::Error parseGraph(llvm::StringRef bytes, RawGraph &graph) {
  Reader in(bytes);
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == GraphName) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "graph.name"))
        return err;
      auto s = in.readBytes();
      if (!s)
        return s.takeError();
      graph.name = s->str();
    } else if (field == GraphNode || field == GraphInitializer ||
               field == GraphInput || field == GraphOutput ||
               field == GraphValueInfo) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "graph field"))
        return err;
      auto inner = in.readBytes();
      if (!inner)
        return inner.takeError();
      if (field == GraphNode) {
        RawNode node;
        if (auto err = parseNode(*inner, node))
          return err;
        graph.nodes.push_back(std::move(node));
      } else if (field == GraphInitializer) {
        RawTensor tensor;
        if (auto err = parseTensor(*inner, tensor))
          return err;
        graph.initializers.push_back(std::move(tensor));
      } else {
        RawValueInfo info;
        if (auto err = parseValueInfo(*inner, info))
          return err;
        if (field == GraphInput)
          graph.inputs.push_back(std::move(info));
        else if (field == GraphOutput)
          graph.outputs.push_back(std::move(info));
        else
          graph.valueInfos.push_back(std::move(info));
      }
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  return llvm::Error::success();
}

static llvm::Expected<RawGraph> parseModel(llvm::StringRef bytes) {
  Reader in(bytes);
  RawGraph graph;
  bool hasGraph = false;
  while (!in.atEnd()) {
    auto keyOr = in.readKey();
    if (!keyOr)
      return keyOr.takeError();
    auto [field, wt] = *keyOr;
    if (field == ModelGraph) {
      if (auto err = expectWire(wt, WireType::LengthDelimited, "model.graph"))
        return err;
      auto inner = in.readBytes();
      if (!inner)
        return inner.takeError();
      if (auto err = parseGraph(*inner, graph))
        return err;
      hasGraph = true;
    } else if (auto err = in.skip(wt)) {
      return err;
    }
  }
  if (!hasGraph)
    return makeError("ONNX model has no graph");
  return graph;
}

static int64_t tensorNumElements(llvm::ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

static llvm::Expected<std::vector<float>>
extractFloats(const RawTensor &tensor, llvm::ArrayRef<int64_t> shape) {
  if (tensor.dataType != 0 && tensor.dataType != OnnxFloat)
    return makeError("initializer '" + tensor.name +
                     "' is not FLOAT; only fp32 is supported");
  const int64_t expected = tensorNumElements(shape);
  if (!tensor.rawData.empty()) {
    if (tensor.rawData.size() != static_cast<size_t>(expected) * sizeof(float))
      return makeError("raw_data size does not match shape for '" + tensor.name +
                       "'");
    std::vector<float> data(static_cast<size_t>(expected));
    std::memcpy(data.data(), tensor.rawData.data(), tensor.rawData.size());
    return data;
  }
  if (!tensor.floatData.empty()) {
    if (static_cast<int64_t>(tensor.floatData.size()) != expected)
      return makeError("float_data count does not match shape for '" +
                       tensor.name + "'");
    return tensor.floatData;
  }
  if (expected == 0)
    return std::vector<float>{};
  return makeError("tensor '" + tensor.name +
                   "' has no float payload (raw_data or float_data)");
}

static llvm::Expected<TensorDesc> tensorToDesc(const RawTensor &tensor) {
  if (tensor.name.empty())
    return makeError("initializer is missing a name");
  auto dataOr = extractFloats(tensor, tensor.dims);
  if (!dataOr)
    return dataOr.takeError();
  TensorDesc desc;
  desc.name = tensor.name;
  desc.shape = tensor.dims;
  desc.data = std::move(*dataOr);
  return desc;
}

static const RawAttribute *findAttr(const RawNode &node, llvm::StringRef name) {
  for (const RawAttribute &attr : node.attributes) {
    if (attr.name == name)
      return &attr;
  }
  return nullptr;
}

static llvm::Expected<std::vector<int64_t>>
inferNodeShape(const NodeInfo &node,
               const llvm::StringMap<std::vector<int64_t>> &shapes) {
  auto lookup = [&](llvm::StringRef name)
      -> llvm::Expected<const std::vector<int64_t> *> {
    auto it = shapes.find(name);
    if (it == shapes.end())
      return makeError("unknown value '" + name + "' while inferring " +
                       node.opType);
    return &it->second;
  };

  if (node.opType == "Add") {
    if (node.inputs.size() != 2)
      return makeError("Add expects 2 inputs");
    auto lhsOr = lookup(node.inputs[0]);
    if (!lhsOr)
      return lhsOr.takeError();
    auto rhsOr = lookup(node.inputs[1]);
    if (!rhsOr)
      return rhsOr.takeError();
    const auto &lhs = **lhsOr;
    const auto &rhs = **rhsOr;
    if (lhs == rhs)
      return lhs;
    if (lhs.empty())
      return rhs;
    if (rhs.empty())
      return lhs;
    return makeError("Add operands must have the same shape or one must be "
                     "a 0-D scalar");
  }

  if (node.opType == "Relu") {
    if (node.inputs.size() != 1)
      return makeError("Relu expects 1 input");
    auto inOr = lookup(node.inputs[0]);
    if (!inOr)
      return inOr.takeError();
    return **inOr;
  }

  if (node.opType == "MatMul") {
    if (node.inputs.size() != 2)
      return makeError("MatMul expects 2 inputs");
    auto lhsOr = lookup(node.inputs[0]);
    if (!lhsOr)
      return lhsOr.takeError();
    auto rhsOr = lookup(node.inputs[1]);
    if (!rhsOr)
      return rhsOr.takeError();
    const auto &lhs = **lhsOr;
    const auto &rhs = **rhsOr;
    if (lhs.size() != 2 || rhs.size() != 2)
      return makeError("MatMul operands must be rank-2");
    if (lhs[1] != rhs[0])
      return makeError("MatMul contracting dimension mismatch");
    return std::vector<int64_t>{lhs[0], rhs[1]};
  }

  return makeError("unsupported op '" + node.opType + "'");
}

static llvm::Error requireValueShape(const RawValueInfo &info) {
  if (info.name.empty())
    return makeError("value_info is missing a name");
  if (info.elemType == 0)
    return makeError("value '" + info.name + "' is missing a type");
  if (info.elemType != OnnxFloat)
    return makeError("only FLOAT (fp32) tensors are supported");
  if (!info.hasShape)
    return makeError("missing tensor shape for '" + info.name + "'");
  return llvm::Error::success();
}

llvm::Expected<ModelInfo> mlir::tc::parseONNXFile(llvm::StringRef path) {
  auto bufOr = llvm::MemoryBuffer::getFile(path);
  if (!bufOr)
    return llvm::createStringError(bufOr.getError(),
                                   "failed to open '" + path + "': " +
                                       bufOr.getError().message());

  auto graphOr = parseModel((*bufOr)->getBuffer());
  if (!graphOr)
    return graphOr.takeError();
  const RawGraph &graph = *graphOr;

  ModelInfo model;
  model.name = graph.name;

  llvm::StringMap<std::vector<int64_t>> shapes;
  llvm::StringSet<> initializerNames;

  for (const RawTensor &init : graph.initializers) {
    auto descOr = tensorToDesc(init);
    if (!descOr)
      return descOr.takeError();
    if (!initializerNames.insert(descOr->name).second)
      return makeError("duplicate initializer '" + descOr->name + "'");
    shapes[descOr->name] = descOr->shape;
    model.initializers.push_back(std::move(*descOr));
  }

  auto addValueInfo = [&](const RawValueInfo &info,
                          std::vector<TensorDesc> *outList,
                          bool skipIfInitializer) -> llvm::Error {
    if (auto err = requireValueShape(info))
      return err;
    auto existing = shapes.find(info.name);
    if (existing != shapes.end() && existing->second != info.shape)
      return makeError("conflicting shapes for '" + info.name + "'");
    shapes[info.name] = info.shape;
    if (skipIfInitializer && initializerNames.contains(info.name))
      return llvm::Error::success();
    if (outList) {
      TensorDesc desc;
      desc.name = info.name;
      desc.shape = info.shape;
      outList->push_back(std::move(desc));
    }
    return llvm::Error::success();
  };

  for (const RawValueInfo &input : graph.inputs) {
    if (auto err = addValueInfo(input, &model.inputs, /*skipIfInitializer=*/true))
      return err;
  }
  for (const RawValueInfo &output : graph.outputs) {
    if (auto err =
            addValueInfo(output, &model.outputs, /*skipIfInitializer=*/false))
      return err;
  }
  for (const RawValueInfo &info : graph.valueInfos) {
    if (auto err = addValueInfo(info, nullptr, /*skipIfInitializer=*/false))
      return err;
  }

  for (const RawNode &node : graph.nodes) {
    if (node.opType.empty())
      return makeError("node is missing op_type");

    if (node.opType == "Constant") {
      if (node.outputs.size() != 1 || node.outputs[0].empty())
        return makeError("Constant node must have one named output");
      const RawAttribute *value = findAttr(node, "value");
      if (!value || !value->hasTensor)
        return makeError("Constant node '" + node.outputs[0] +
                         "' is missing a tensor 'value' attribute");
      RawTensor tensor = value->tensor;
      if (tensor.name.empty())
        tensor.name = node.outputs[0];
      auto descOr = tensorToDesc(tensor);
      if (!descOr)
        return descOr.takeError();
      descOr->name = node.outputs[0];
      if (!initializerNames.insert(descOr->name).second)
        return makeError("duplicate constant '" + descOr->name + "'");
      shapes[descOr->name] = descOr->shape;
      model.initializers.push_back(std::move(*descOr));
      continue;
    }

    if (node.opType != "Add" && node.opType != "Relu" && node.opType != "MatMul")
      return makeError("unsupported op '" + node.opType +
                       "'; supported: Constant, Add, Relu, MatMul");

    NodeInfo info;
    info.opType = node.opType;
    for (const std::string &in : node.inputs) {
      if (in.empty())
        return makeError(info.opType + " has an empty input name");
      info.inputs.push_back(in);
    }
    for (const std::string &out : node.outputs) {
      if (out.empty())
        return makeError(info.opType + " has an empty output name");
      info.outputs.push_back(out);
    }
    if (info.outputs.size() != 1)
      return makeError(info.opType + " must have exactly one output");

    auto inferredOr = inferNodeShape(info, shapes);
    if (!inferredOr)
      return inferredOr.takeError();
    auto existing = shapes.find(info.outputs[0]);
    if (existing != shapes.end() && existing->second != *inferredOr)
      return makeError("inferred shape for '" + info.outputs[0] +
                       "' conflicts with declared shape");
    shapes[info.outputs[0]] = *inferredOr;
    model.nodes.push_back(std::move(info));
  }

  if (model.outputs.empty())
    return makeError("graph has no outputs");
  for (const TensorDesc &out : model.outputs) {
    if (!shapes.contains(out.name))
      return makeError("output '" + out.name + "' is not produced by the graph");
  }

  return model;
}
