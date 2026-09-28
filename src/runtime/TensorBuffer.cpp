//===- TensorBuffer.cpp - Host-side fp32 tensors ------------------*- C++ -*-===//

#include "runtime/TensorBuffer.h"

#include "support/Common.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstring>

using namespace mlir::tc;
using llvm::StringRef;

TensorBuffer mlir::tc::makeSequentialTensor(llvm::ArrayRef<int64_t> shape) {
  TensorBuffer buf;
  buf.shape.assign(shape.begin(), shape.end());
  buf.data.resize(static_cast<size_t>(numElements(shape)));
  for (size_t i = 0, e = buf.data.size(); i < e; ++i)
    buf.data[i] = static_cast<float>(i);
  return buf;
}

void mlir::tc::printTensorBuffer(const TensorBuffer &buffer,
                                 llvm::raw_ostream &os) {
  os << "shape:";
  if (buffer.shape.empty())
    os << " scalar";
  else
    llvm::interleave(buffer.shape, os, "x");
  os << "\n";

  size_t rowLength = buffer.shape.empty() ? 1 : buffer.shape.back();
  for (size_t i = 0, e = buffer.data.size(); i < e; ++i) {
    if (i && rowLength > 0 && i % rowLength == 0)
      os << "\n";
    else if (i)
      os << " ";
    os << buffer.data[i];
  }
  os << "\n";
}

/// Parse the `'shape': (2, 3)` entry of an .npy header dict.
static llvm::Expected<std::vector<int64_t>> parseNpyShape(StringRef header) {
  size_t key = header.find("'shape':");
  if (key == StringRef::npos)
    key = header.find("\"shape\":");
  size_t open = header.find('(', key);
  size_t close = header.find(')', open);
  if (key == StringRef::npos || open == StringRef::npos ||
      close == StringRef::npos)
    return makeError("missing shape");

  std::vector<int64_t> shape;
  StringRef dims = header.slice(open + 1, close);
  while (!dims.empty()) {
    auto [dim, rest] = dims.split(',');
    dims = rest;
    dim = dim.trim();
    if (dim.empty())
      continue;
    int64_t value;
    if (dim.getAsInteger(10, value))
      return makeError("invalid shape dim '" + dim + "'");
    shape.push_back(value);
  }
  return shape;
}

llvm::Expected<TensorBuffer> mlir::tc::loadNpyF32(StringRef path) {
  auto fail = [&](const llvm::Twine &msg) {
    return makeError("'" + path + "': " + msg);
  };

  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return fail(file.getError().message());
  StringRef bytes = (*file)->getBuffer();

  // Layout: "\x93NUMPY", major, minor, header length (u16 for v1, u32 for
  // v2/v3), ASCII header dict, raw data.
  if (!bytes.starts_with("\x93NUMPY") || bytes.size() < 10)
    return fail("not a .npy file");
  uint8_t major = static_cast<uint8_t>(bytes[6]);
  if (major < 1 || major > 3)
    return fail("unsupported .npy version");
  size_t lenBytes = major == 1 ? 2 : 4;
  uint32_t headerLen = 0;
  std::memcpy(&headerLen, bytes.data() + 8, lenBytes);
  size_t headerStart = 8 + lenBytes;
  if (bytes.size() < headerStart + headerLen)
    return fail("truncated .npy header");
  StringRef header = bytes.substr(headerStart, headerLen);

  if (!header.contains("'descr': '<f4'") &&
      !header.contains("\"descr\": \"<f4\"") &&
      !header.contains("'descr': '|f4'"))
    return fail("must be little-endian float32");
  if (header.contains("'fortran_order': True") ||
      header.contains("\"fortran_order\": true"))
    return fail("must be C-contiguous");

  llvm::Expected<std::vector<int64_t>> shape = parseNpyShape(header);
  if (!shape)
    return fail(llvm::toString(shape.takeError()));

  TensorBuffer buf;
  buf.shape = std::move(*shape);
  buf.data.resize(static_cast<size_t>(numElements(buf.shape)));
  StringRef payload = bytes.drop_front(headerStart + headerLen);
  size_t payloadBytes = buf.data.size() * sizeof(float);
  if (payload.size() < payloadBytes)
    return fail("truncated payload");
  std::memcpy(buf.data.data(), payload.data(), payloadBytes);
  return buf;
}
