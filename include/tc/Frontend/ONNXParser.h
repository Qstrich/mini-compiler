//===- ONNXParser.h - Parse ONNX protobuf into ModelInfo ----------*- C++ -*-===//

#ifndef TC_FRONTEND_ONNXPARSER_H
#define TC_FRONTEND_ONNXPARSER_H

#include "tc/Frontend/ModelInfo.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace tc {

/// Parse a `.onnx` file into ModelInfo. Rejects dynamic shapes, non-fp32,
/// and ops other than Constant / Add / Relu / MatMul.
llvm::Expected<ModelInfo> parseONNXFile(llvm::StringRef path);

} // namespace tc
} // namespace mlir

#endif // TC_FRONTEND_ONNXPARSER_H
