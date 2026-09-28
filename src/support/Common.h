//===- Common.h - Small helpers shared across the compiler -------*- C++ -*-===//

#ifndef TC_SUPPORT_COMMON_H
#define TC_SUPPORT_COMMON_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir {
namespace tc {

/// A plain string error for `llvm::Expected` / `llvm::Error` returns.
inline llvm::Error makeError(const llvm::Twine &msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
}

/// Number of elements in a static shape (1 for a 0-D scalar).
inline int64_t numElements(llvm::ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

} // namespace tc
} // namespace mlir

#endif // TC_SUPPORT_COMMON_H
