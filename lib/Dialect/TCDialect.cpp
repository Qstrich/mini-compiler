//===- TCDialect.cpp - TC dialect -------------------------------*- C++ -*-===//

#include "tc/Dialect/TCDialect.h"
#include "tc/Dialect/TCOps.h"

using namespace mlir;
using namespace mlir::tc;

#include "tc/Dialect/TCOpsDialect.cpp.inc"

void TCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tc/Dialect/TCOps.cpp.inc"
      >();
}
