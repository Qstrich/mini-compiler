//===- OnnxProto.cpp - Minimal protobuf decoder for ONNX ----------*- C++ -*-===//
//
// Just enough of the protobuf wire format to decode the ONNX subset in
// OnnxProto.h without linking libprotobuf. Each `parseX` below is a switch
// over the field numbers of one message in third_party/onnx/onnx.proto;
// unknown fields are skipped.
//
//===----------------------------------------------------------------------===//

#include "import/OnnxProto.h"

#include "support/Common.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/bit.h"

#include <cstring>

using namespace mlir::tc;
using namespace mlir::tc::onnx;
using llvm::Error;
using llvm::Expected;
using llvm::StringRef;

namespace {

enum class WireType : uint32_t {
  Varint = 0,
  Fixed64 = 1,
  LengthDelimited = 2,
  Fixed32 = 5,
};

Error decodeError(const llvm::Twine &msg) {
  return makeError("onnx protobuf: " + msg);
}

/// Cursor over the bytes of one message.
class Reader {
public:
  explicit Reader(StringRef data) : data(data) {}

  bool atEnd() const { return pos >= data.size(); }

  Expected<uint64_t> readVarint() {
    uint64_t result = 0;
    for (unsigned shift = 0; pos < data.size(); shift += 7) {
      if (shift > 63)
        return decodeError("varint overflow");
      uint8_t byte = static_cast<uint8_t>(data[pos++]);
      result |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0)
        return result;
    }
    return decodeError("truncated varint");
  }

  Expected<uint32_t> readFixed32() {
    if (pos + 4 > data.size())
      return decodeError("truncated fixed32");
    uint32_t value;
    std::memcpy(&value, data.data() + pos, 4);
    pos += 4;
    return value;
  }

  Expected<StringRef> readBytes() {
    Expected<uint64_t> len = readVarint();
    if (!len)
      return len.takeError();
    if (*len > data.size() - pos)
      return decodeError("truncated length-delimited field");
    StringRef bytes = data.substr(pos, *len);
    pos += *len;
    return bytes;
  }

  Error skip(WireType type) {
    switch (type) {
    case WireType::Varint:
      return readVarint().takeError();
    case WireType::Fixed64:
      if (pos + 8 > data.size())
        return decodeError("truncated fixed64");
      pos += 8;
      return Error::success();
    case WireType::LengthDelimited:
      return readBytes().takeError();
    case WireType::Fixed32:
      return readFixed32().takeError();
    }
    return decodeError("unknown wire type");
  }

private:
  StringRef data;
  size_t pos = 0;
};

/// One field of a message: its number, wire type, and a reader positioned at
/// its payload. Every method checks the wire type and consumes the payload.
class Field {
public:
  Field(uint32_t number, WireType wireType, Reader &in)
      : num(number), wt(wireType), in(in) {}

  uint32_t number() const { return num; }
  WireType wireType() const { return wt; }

  Error skip() { return in.skip(wt); }

  template <typename IntT>
  Error readInt(IntT &out) {
    if (Error err = expect(WireType::Varint))
      return err;
    Expected<uint64_t> v = in.readVarint();
    if (!v)
      return v.takeError();
    out = static_cast<IntT>(*v);
    return Error::success();
  }

  Expected<uint32_t> readFixed32() {
    if (Error err = expect(WireType::Fixed32))
      return std::move(err);
    return in.readFixed32();
  }

  Expected<StringRef> readBytes() {
    if (Error err = expect(WireType::LengthDelimited))
      return std::move(err);
    return in.readBytes();
  }

  Error readString(std::string &out) {
    Expected<StringRef> bytes = readBytes();
    if (!bytes)
      return bytes.takeError();
    out = bytes->str();
    return Error::success();
  }

  /// Decode a nested message with `parse`.
  template <typename T>
  Error readMessage(Error (*parse)(StringRef, T &), T &out) {
    Expected<StringRef> bytes = readBytes();
    if (!bytes)
      return bytes.takeError();
    return parse(*bytes, out);
  }

private:
  Error expect(WireType want) {
    if (wt == want)
      return Error::success();
    return decodeError("unexpected wire type for field " + llvm::Twine(num));
  }

  uint32_t num;
  WireType wt;
  Reader &in;
};

/// Call `fn` for every field of the message in `bytes`.
Error forEachField(StringRef bytes, llvm::function_ref<Error(Field)> fn) {
  Reader in(bytes);
  while (!in.atEnd()) {
    Expected<uint64_t> key = in.readVarint();
    if (!key)
      return key.takeError();
    Field field(static_cast<uint32_t>(*key >> 3),
                static_cast<WireType>(*key & 0x7), in);
    if (Error err = fn(field))
      return err;
  }
  return Error::success();
}

/// `repeated float float_data = 4 [packed = true]`; also accepts the unpacked
/// encoding (one fixed32 per element).
Error readFloats(Field &f, std::vector<float> &out) {
  if (f.wireType() == WireType::Fixed32) {
    Expected<uint32_t> bits = f.readFixed32();
    if (!bits)
      return bits.takeError();
    out.push_back(llvm::bit_cast<float>(*bits));
    return Error::success();
  }
  Expected<StringRef> packed = f.readBytes();
  if (!packed)
    return packed.takeError();
  if (packed->size() % sizeof(float) != 0)
    return decodeError("packed float_data size is not a multiple of 4");
  size_t old = out.size();
  out.resize(old + packed->size() / sizeof(float));
  std::memcpy(out.data() + old, packed->data(), packed->size());
  return Error::success();
}

//===----------------------------------------------------------------------===//
// One function per message, innermost first.
//===----------------------------------------------------------------------===//

Error parseDimension(StringRef bytes, Dimension &dim) {
  return forEachField(bytes, [&](Field f) {
    switch (f.number()) {
    case 1: // dim_value
      return f.readInt(dim.value.emplace());
    case 2: // dim_param
      return f.readString(dim.param);
    default:
      return f.skip();
    }
  });
}

Error parseShape(StringRef bytes, std::vector<Dimension> &shape) {
  return forEachField(bytes, [&](Field f) {
    if (f.number() == 1) // dim
      return f.readMessage(parseDimension, shape.emplace_back());
    return f.skip();
  });
}

/// TypeProto.Tensor
Error parseTensorType(StringRef bytes, ValueInfoProto &info) {
  return forEachField(bytes, [&](Field f) {
    switch (f.number()) {
    case 1: // elem_type
      return f.readInt(info.elemType);
    case 2: // shape
      return f.readMessage(parseShape, info.shape.emplace());
    default:
      return f.skip();
    }
  });
}

/// TypeProto (only the tensor_type case of the oneof).
Error parseType(StringRef bytes, ValueInfoProto &info) {
  return forEachField(bytes, [&](Field f) {
    if (f.number() == 1) // tensor_type
      return f.readMessage(parseTensorType, info);
    return f.skip();
  });
}

Error parseValueInfo(StringRef bytes, ValueInfoProto &info) {
  return forEachField(bytes, [&](Field f) {
    switch (f.number()) {
    case 1: // name
      return f.readString(info.name);
    case 2: // type
      return f.readMessage(parseType, info);
    default:
      return f.skip();
    }
  });
}

Error parseTensor(StringRef bytes, TensorProto &tensor) {
  return forEachField(bytes, [&](Field f) {
    switch (f.number()) {
    case 1: // dims
      return f.readInt(tensor.dims.emplace_back());
    case 2: // data_type
      return f.readInt(tensor.dataType);
    case 4: // float_data
      return readFloats(f, tensor.floatData);
    case 8: // name
      return f.readString(tensor.name);
    case 9: // raw_data
      return f.readString(tensor.rawData);
    default:
      return f.skip();
    }
  });
}

Error parseAttribute(StringRef bytes, AttributeProto &attr) {
  return forEachField(bytes, [&](Field f) {
    switch (f.number()) {
    case 1: // name
      return f.readString(attr.name);
    case 5: // t
      return f.readMessage(parseTensor, attr.t.emplace());
    default:
      return f.skip();
    }
  });
}

Error parseNode(StringRef bytes, NodeProto &node) {
  return forEachField(bytes, [&](Field f) {
    switch (f.number()) {
    case 1: // input
      return f.readString(node.inputs.emplace_back());
    case 2: // output
      return f.readString(node.outputs.emplace_back());
    case 3: // name
      return f.readString(node.name);
    case 4: // op_type
      return f.readString(node.opType);
    case 5: // attribute
      return f.readMessage(parseAttribute, node.attributes.emplace_back());
    default:
      return f.skip();
    }
  });
}

Error parseGraph(StringRef bytes, GraphProto &graph) {
  return forEachField(bytes, [&](Field f) {
    switch (f.number()) {
    case 1: // node
      return f.readMessage(parseNode, graph.nodes.emplace_back());
    case 2: // name
      return f.readString(graph.name);
    case 5: // initializer
      return f.readMessage(parseTensor, graph.initializers.emplace_back());
    case 11: // input
      return f.readMessage(parseValueInfo, graph.inputs.emplace_back());
    case 12: // output
      return f.readMessage(parseValueInfo, graph.outputs.emplace_back());
    default:
      return f.skip();
    }
  });
}

} // namespace

Expected<GraphProto> onnx::decodeModel(StringRef bytes) {
  std::optional<GraphProto> graph;
  if (Error err = forEachField(bytes, [&](Field f) {
        if (f.number() == 7) // ModelProto.graph
          return f.readMessage(parseGraph, graph.emplace());
        return f.skip();
      }))
    return std::move(err);
  if (!graph)
    return decodeError("model has no graph");
  return std::move(*graph);
}
