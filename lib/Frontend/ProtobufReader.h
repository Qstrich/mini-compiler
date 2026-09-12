//===- ProtobufReader.h - Minimal proto2 wire decoder -------------*- C++ -*-===//
//
// Enough of the protobuf wire format to parse the ONNX subset in
// third_party/onnx/onnx.proto. Unknown fields are skipped.

#ifndef TC_FRONTEND_PROTOBUFREADER_H
#define TC_FRONTEND_PROTOBUFREADER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mlir {
namespace tc {
namespace proto {

enum class WireType : uint32_t {
  Varint = 0,
  Fixed64 = 1,
  LengthDelimited = 2,
  Fixed32 = 5,
};

class Reader {
public:
  explicit Reader(llvm::StringRef data) : data(data), pos(0) {}

  bool atEnd() const { return pos >= data.size(); }

  llvm::Expected<uint64_t> readVarint() {
    uint64_t result = 0;
    unsigned shift = 0;
    while (pos < data.size()) {
      uint8_t byte = static_cast<uint8_t>(data[pos++]);
      result |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0)
        return result;
      shift += 7;
      if (shift > 63)
        return error("varint overflow");
    }
    return error("truncated varint");
  }

  llvm::Expected<uint32_t> readFixed32() {
    if (pos + 4 > data.size())
      return error("truncated fixed32");
    uint32_t value = 0;
    std::memcpy(&value, data.data() + pos, 4);
    pos += 4;
    return value;
  }

  llvm::Expected<llvm::StringRef> readBytes() {
    auto lenOr = readVarint();
    if (!lenOr)
      return lenOr.takeError();
    if (pos + *lenOr > data.size())
      return error("truncated length-delimited field");
    llvm::StringRef bytes = data.substr(pos, *lenOr);
    pos += *lenOr;
    return bytes;
  }

  llvm::Expected<std::pair<uint32_t, WireType>> readKey() {
    auto keyOr = readVarint();
    if (!keyOr)
      return keyOr.takeError();
    uint32_t field = static_cast<uint32_t>(*keyOr >> 3);
    auto wt = static_cast<WireType>(*keyOr & 0x7);
    return std::make_pair(field, wt);
  }

  llvm::Error skip(WireType type) {
    switch (type) {
    case WireType::Varint: {
      auto v = readVarint();
      return v ? llvm::Error::success() : v.takeError();
    }
    case WireType::Fixed64:
      if (pos + 8 > data.size())
        return error("truncated fixed64");
      pos += 8;
      return llvm::Error::success();
    case WireType::LengthDelimited: {
      auto b = readBytes();
      return b ? llvm::Error::success() : b.takeError();
    }
    case WireType::Fixed32: {
      auto v = readFixed32();
      return v ? llvm::Error::success() : v.takeError();
    }
    }
    return error("unknown wire type");
  }

private:
  static llvm::Error error(const llvm::Twine &msg) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "onnx protobuf: " + msg);
  }

  llvm::StringRef data;
  size_t pos;
};

} // namespace proto
} // namespace tc
} // namespace mlir

#endif // TC_FRONTEND_PROTOBUFREADER_H
