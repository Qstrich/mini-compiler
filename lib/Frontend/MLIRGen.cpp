//===- MLIRGen.cpp - ModelInfo -> tc dialect ----------------------*- C++ -*-===//

#include "tc/Frontend/MLIRGen.h"

#include "tc/Dialect/TCDialect.h"
#include "tc/Dialect/TCOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

using namespace mlir;
using namespace mlir::tc;

static llvm::Error makeError(const llvm::Twine &msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
}

static RankedTensorType tensorType(MLIRContext &context,
                                   llvm::ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, Float32Type::get(&context));
}

llvm::Expected<OwningOpRef<ModuleOp>>
mlir::tc::generateMLIR(MLIRContext &context, const ModelInfo &model,
                       llvm::StringRef filename) {
  context.loadDialect<TCDialect, func::FuncDialect>();

  Location loc = FileLineColLoc::get(&context, filename, /*line=*/0,
                                     /*column=*/0);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(module->getBodyRegion());

  SmallVector<Type> argTypes;
  argTypes.reserve(model.inputs.size());
  for (const TensorDesc &input : model.inputs)
    argTypes.push_back(tensorType(context, input.shape));

  SmallVector<Type> resultTypes;
  resultTypes.reserve(model.outputs.size());
  for (const TensorDesc &output : model.outputs)
    resultTypes.push_back(tensorType(context, output.shape));

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  auto func = func::FuncOp::create(builder, loc, "main", funcType);
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::StringMap<Value> values;
  for (auto [i, input] : llvm::enumerate(model.inputs))
    values[input.name] = entry->getArgument(i);

  for (const TensorDesc &init : model.initializers) {
    auto type = tensorType(context, init.shape);
    auto attr = DenseElementsAttr::get(type, llvm::ArrayRef<float>(init.data));
    auto op = ConstantOp::create(builder, loc, attr);
    values[init.name] = op.getResult();
  }

  auto lookup = [&](llvm::StringRef name) -> llvm::Expected<Value> {
    auto it = values.find(name);
    if (it == values.end())
      return makeError("unknown value '" + name + "'");
    return it->second;
  };

  for (const NodeInfo &node : model.nodes) {
    if (node.outputs.size() != 1)
      return makeError(node.opType + " must have exactly one output");

    if (node.opType == "Add") {
      if (node.inputs.size() != 2)
        return makeError("Add expects 2 inputs");
      auto lhsOr = lookup(node.inputs[0]);
      if (!lhsOr)
        return lhsOr.takeError();
      auto rhsOr = lookup(node.inputs[1]);
      if (!rhsOr)
        return rhsOr.takeError();
      auto lhsType = dyn_cast<RankedTensorType>(lhsOr->getType());
      auto rhsType = dyn_cast<RankedTensorType>(rhsOr->getType());
      if (!lhsType || !rhsType)
        return makeError("Add operands must be ranked tensors");
      Type resultType = lhsType.getRank() == 0 ? Type(rhsType) : Type(lhsType);
      auto op = AddOp::create(builder, loc, resultType, *lhsOr, *rhsOr);
      values[node.outputs[0]] = op.getResult();
      continue;
    }

    if (node.opType == "Relu") {
      if (node.inputs.size() != 1)
        return makeError("Relu expects 1 input");
      auto inOr = lookup(node.inputs[0]);
      if (!inOr)
        return inOr.takeError();
      auto op = ReluOp::create(builder, loc, inOr->getType(), *inOr);
      values[node.outputs[0]] = op.getResult();
      continue;
    }

    if (node.opType == "MatMul") {
      if (node.inputs.size() != 2)
        return makeError("MatMul expects 2 inputs");
      auto lhsOr = lookup(node.inputs[0]);
      if (!lhsOr)
        return lhsOr.takeError();
      auto rhsOr = lookup(node.inputs[1]);
      if (!rhsOr)
        return rhsOr.takeError();
      auto lhsType = dyn_cast<RankedTensorType>(lhsOr->getType());
      auto rhsType = dyn_cast<RankedTensorType>(rhsOr->getType());
      if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
          rhsType.getRank() != 2)
        return makeError("MatMul operands must be rank-2 tensors");
      auto resultType = RankedTensorType::get(
          {lhsType.getDimSize(0), rhsType.getDimSize(1)},
          lhsType.getElementType());
      auto op = MatMulOp::create(builder, loc, resultType, *lhsOr, *rhsOr);
      values[node.outputs[0]] = op.getResult();
      continue;
    }

    return makeError("unsupported op '" + node.opType + "' in MLIRGen");
  }

  SmallVector<Value> results;
  results.reserve(model.outputs.size());
  for (const TensorDesc &output : model.outputs) {
    auto valOr = lookup(output.name);
    if (!valOr)
      return valOr.takeError();
    results.push_back(*valOr);
  }
  func::ReturnOp::create(builder, loc, results);

  if (failed(verify(*module)))
    return makeError("generated MLIR failed verification");
  return module;
}
