//===- TCOps.cpp - TC dialect ops -------------------------------*- C++ -*-===//

#include "tc/Dialect/TCOps.h"
#include "tc/Dialect/TCDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "tc/Dialect/TCOps.cpp.inc"

using namespace mlir;
using namespace mlir::tc;

static bool isF32Ranked(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getElementType().isF32();
}

static bool isScalarF32Tensor(RankedTensorType type) {
  return type.getRank() == 0 && type.getElementType().isF32();
}

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }

LogicalResult ConstantOp::verify() {
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType || !resultType.getElementType().isF32())
    return emitOpError("result must be a ranked tensor of f32");

  auto attrType = dyn_cast<RankedTensorType>(getValue().getType());
  if (!attrType || attrType != resultType)
    return emitOpError("value attribute type must match result type");
  return success();
}

LogicalResult AddOp::verify() {
  auto lhs = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhs = dyn_cast<RankedTensorType>(getRhs().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhs || !rhs || !result || !isF32Ranked(lhs) || !isF32Ranked(rhs) ||
      !isF32Ranked(result))
    return emitOpError("operands and result must be ranked f32 tensors");

  RankedTensorType expected;
  if (lhs == rhs) {
    expected = lhs;
  } else if (isScalarF32Tensor(lhs)) {
    expected = rhs;
  } else if (isScalarF32Tensor(rhs)) {
    expected = lhs;
  } else {
    return emitOpError(
        "operands must have the same shape or one must be a 0-D scalar");
  }

  if (result != expected)
    return emitOpError("result type must match the non-scalar operand type");
  return success();
}

LogicalResult ReluOp::verify() {
  if (!isF32Ranked(getInput().getType()))
    return emitOpError("input must be a ranked f32 tensor");
  return success();
}

LogicalResult MatMulOp::verify() {
  auto lhs = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhs = dyn_cast<RankedTensorType>(getRhs().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhs || !rhs || !result || !isF32Ranked(lhs) || !isF32Ranked(rhs) ||
      !isF32Ranked(result))
    return emitOpError("operands and result must be ranked f32 tensors");

  if (lhs.getRank() != 2 || rhs.getRank() != 2)
    return emitOpError("operands must be rank-2 tensors");
  if (result.getRank() != 2)
    return emitOpError("result must be a rank-2 tensor");

  int64_t m = lhs.getDimSize(0);
  int64_t k = lhs.getDimSize(1);
  int64_t kRhs = rhs.getDimSize(0);
  int64_t n = rhs.getDimSize(1);
  if (k != kRhs)
    return emitOpError("contracting dimension mismatch: lhs K=")
           << k << " vs rhs K=" << kRhs;
  if (result.getDimSize(0) != m || result.getDimSize(1) != n)
    return emitOpError("result must have shape ") << m << "x" << n;
  return success();
}
