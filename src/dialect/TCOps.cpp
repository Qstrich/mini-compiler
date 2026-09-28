//===- TCOps.cpp - TC dialect ops -------------------------------*- C++ -*-===//
//
// Shape rules for the tc ops live here and nowhere else. `tc.add` and
// `tc.matmul` implement InferTypeOpInterface: the frontend uses
// `inferReturnTypes` to build ops, and the interface's verifier rejects any
// op whose declared result type differs from the inferred one.
//
//===----------------------------------------------------------------------===//

#include "dialect/TCOps.h"
#include "dialect/TCDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#define GET_OP_CLASSES
#include "dialect/TCOps.cpp.inc"

using namespace mlir;
using namespace mlir::tc;

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }

/// Same shape, or one operand is a 0-D scalar that broadcasts.
LogicalResult AddOp::inferReturnTypes(MLIRContext *, std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type> &inferred) {
  auto lhs = cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhs = cast<RankedTensorType>(adaptor.getRhs().getType());
  if (lhs == rhs || rhs.getRank() == 0)
    inferred.push_back(lhs);
  else if (lhs.getRank() == 0)
    inferred.push_back(rhs);
  else
    return emitOptionalError(
        loc, "operands must have the same shape or one must be a 0-D scalar");
  return success();
}

/// A[M,K] @ B[K,N] -> C[M,N].
LogicalResult MatMulOp::inferReturnTypes(MLIRContext *,
                                         std::optional<Location> loc,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type> &inferred) {
  auto lhs = cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhs = cast<RankedTensorType>(adaptor.getRhs().getType());
  if (lhs.getRank() != 2 || rhs.getRank() != 2)
    return emitOptionalError(loc, "operands must be rank-2 tensors");
  if (lhs.getDimSize(1) != rhs.getDimSize(0))
    return emitOptionalError(loc, "contracting dimension mismatch: lhs K=",
                             lhs.getDimSize(1), " vs rhs K=",
                             rhs.getDimSize(0));
  inferred.push_back(RankedTensorType::get(
      {lhs.getDimSize(0), rhs.getDimSize(1)}, lhs.getElementType()));
  return success();
}
