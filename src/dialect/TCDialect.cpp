//===- TCDialect.cpp - TC dialect -------------------------------*- C++ -*-===//

#include "dialect/TCDialect.h"
#include "dialect/TCOps.h"

using namespace mlir;
using namespace mlir::tc;

#include "dialect/TCOpsDialect.cpp.inc"

void TCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialect/TCOps.cpp.inc"
      >();
}
