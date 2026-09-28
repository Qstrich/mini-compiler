//===- Passes.h - TC conversion passes ---------------------------*- C++ -*-===//

#ifndef TC_LOWERING_PASSES_H
#define TC_LOWERING_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace tc {

#define GEN_PASS_DECL
#include "lowering/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "lowering/Passes.h.inc"

} // namespace tc
} // namespace mlir

#endif // TC_LOWERING_PASSES_H
