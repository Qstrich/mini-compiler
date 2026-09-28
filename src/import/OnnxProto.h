//===- OnnxProto.h - Decoded ONNX protobuf messages ---------------*- C++ -*-===//
//
// Plain structs mirroring the subset of third_party/onnx/onnx.proto that the
// importer reads, plus a decoder that fills them from protobuf wire bytes.
// Decoding is purely syntactic; ONNXImporter.cpp decides what is supported.
//
//===----------------------------------------------------------------------===//

#ifndef TC_IMPORT_ONNXPROTO_H
#define TC_IMPORT_ONNXPROTO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace tc {
namespace onnx {

/// TensorProto.DataType.FLOAT
constexpr int32_t kFloat = 1;

struct TensorProto {
  std::string name;
  std::vector<int64_t> dims;
  int32_t dataType = 0;
  std::vector<float> floatData;
  std::string rawData;
};

struct AttributeProto {
  std::string name;
  std::optional<TensorProto> t;
};

struct NodeProto {
  std::string name;
  std::string opType;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<AttributeProto> attributes;
};

/// TensorShapeProto.Dimension: either a static value or a symbolic name.
struct Dimension {
  std::optional<int64_t> value;
  std::string param;
};

/// ValueInfoProto with its TypeProto.Tensor flattened in.
struct ValueInfoProto {
  std::string name;
  int32_t elemType = 0;
  std::optional<std::vector<Dimension>> shape;
};

struct GraphProto {
  std::string name;
  std::vector<NodeProto> nodes;
  std::vector<TensorProto> initializers;
  std::vector<ValueInfoProto> inputs;
  std::vector<ValueInfoProto> outputs;
};

/// Decode a serialized ModelProto and return its graph.
llvm::Expected<GraphProto> decodeModel(llvm::StringRef bytes);

} // namespace onnx
} // namespace tc
} // namespace mlir

#endif // TC_IMPORT_ONNXPROTO_H
