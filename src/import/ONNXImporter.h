//===- ONNXImporter.h - Import an ONNX file as ModelInfo ----------*- C++ -*-===//
//
// Stage: import, step 1 of 2.
//   In:   path to a .onnx file.
//   Out:  ModelInfo, a plain C++ view of the graph (static fp32 tensors,
//         weights, nodes). Next: generateMLIR (import/MLIRGen.h).
//
//===----------------------------------------------------------------------===//

#ifndef TC_IMPORT_ONNXIMPORTER_H
#define TC_IMPORT_ONNXIMPORTER_H

#include "import/ModelInfo.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace tc {

/// Read a `.onnx` file into ModelInfo. Rejects dynamic shapes and non-fp32
/// tensors; op types are checked later by generateMLIR.
llvm::Expected<ModelInfo> parseONNXFile(llvm::StringRef path);

} // namespace tc
} // namespace mlir

#endif // TC_IMPORT_ONNXIMPORTER_H
