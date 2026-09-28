//===- TCOps.h - TC dialect ops ---------------------------------*- C++ -*-===//
//
// Stage: dialect. The tc IR that import produces and lowering consumes:
// tc.constant, tc.add, tc.relu, tc.matmul. Ops are declared in TCOps.td;
// their shape rules (inferReturnTypes) live in TCOps.cpp and nowhere else.
//
//===----------------------------------------------------------------------===//

#ifndef TC_TCOPS_H
#define TC_TCOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "dialect/TCOps.h.inc"

#endif // TC_TCOPS_H
