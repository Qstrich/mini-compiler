//===- Passes.h - TC conversion passes ---------------------------*- C++ -*-===//

#ifndef TC_CONVERSION_PASSES_H
#define TC_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace tc {

#define GEN_PASS_DECL
#include "tc/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tc/Conversion/Passes.h.inc"

} // namespace tc
} // namespace mlir

#endif // TC_CONVERSION_PASSES_H
