//===- MLIRGen.cpp - ModelInfo -> tc dialect ----------------------*- C++ -*-===//
//
// Emits `func.func @main` with one tc op per ONNX node. This file is the only
// place that maps ONNX op names to tc ops; result types come from the ops'
// own inference (TCOps.cpp), so no shape logic lives here.
//
//===----------------------------------------------------------------------===//

#include "import/MLIRGen.h"

#include "dialect/TCDialect.h"
#include "dialect/TCOps.h"
#include "support/Common.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::tc;
using llvm::Expected;

/// Create `OpTy`, letting the dialect infer its result type. The generated
/// inferring builder aborts on invalid operands, so infer first and turn a
/// failure into an llvm::Error (the diagnostic is already emitted at `loc`).
template <typename OpTy>
static Expected<Value> createInferred(OpBuilder &builder, Location loc,
                                      ValueRange operands) {
  SmallVector<Type> resultTypes;
  if (failed(OpTy::inferReturnTypes(
          builder.getContext(), loc, operands, DictionaryAttr(),
          PropertyRef(), RegionRange(), resultTypes)))
    return makeError("invalid operands for " + OpTy::getOperationName());
  return OpTy::create(builder, loc, resultTypes, operands).getResult();
}

namespace {
struct OpSpec {
  llvm::StringLiteral onnxName;
  unsigned numInputs;
  Expected<Value> (*build)(OpBuilder &, Location, ValueRange);
};
} // namespace

/// ONNX ops lowered to tc ops. (`Constant` is folded into initializers by the
/// importer.)
static const OpSpec kSupportedOps[] = {
    {"Add", 2, createInferred<AddOp>},
    {"Relu", 1, createInferred<ReluOp>},
    {"MatMul", 2, createInferred<MatMulOp>},
};

static const OpSpec *findOp(llvm::StringRef onnxName) {
  for (const OpSpec &spec : kSupportedOps)
    if (spec.onnxName == onnxName)
      return &spec;
  return nullptr;
}

static SmallVector<Type> tensorTypes(MLIRContext &context,
                                     llvm::ArrayRef<TensorDesc> tensors) {
  return llvm::map_to_vector(tensors, [&](const TensorDesc &t) -> Type {
    return RankedTensorType::get(t.shape, Float32Type::get(&context));
  });
}

Expected<OwningOpRef<ModuleOp>>
mlir::tc::generateMLIR(MLIRContext &context, const ModelInfo &model,
                       llvm::StringRef filename) {
  context.loadDialect<TCDialect, func::FuncDialect>();

  Location fileLoc = FileLineColLoc::get(&context, filename, 0, 0);
  auto locFor = [&](llvm::StringRef valueName) -> Location {
    return NameLoc::get(StringAttr::get(&context, valueName), fileLoc);
  };

  OwningOpRef<ModuleOp> module = ModuleOp::create(fileLoc);
  OpBuilder builder(module->getBodyRegion());

  auto funcType = builder.getFunctionType(tensorTypes(context, model.inputs),
                                          tensorTypes(context, model.outputs));
  auto func = func::FuncOp::create(builder, fileLoc, "main", funcType);
  // The JIT calls @main through its C wrapper, `_mlir_ciface_main`.
  func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                builder.getUnitAttr());
  builder.setInsertionPointToStart(func.addEntryBlock());

  // ONNX value name -> SSA value.
  llvm::StringMap<Value> values;
  for (auto [input, arg] : llvm::zip(model.inputs, func.getArguments()))
    values[input.name] = arg;

  for (const TensorDesc &init : model.initializers) {
    auto type = RankedTensorType::get(init.shape, builder.getF32Type());
    auto attr = DenseElementsAttr::get(type, llvm::ArrayRef<float>(init.data));
    values[init.name] =
        ConstantOp::create(builder, locFor(init.name), attr).getResult();
  }

  auto lookup = [&](llvm::StringRef name) -> Expected<Value> {
    auto it = values.find(name);
    if (it == values.end())
      return makeError("unknown value '" + name + "'");
    return it->second;
  };

  for (const NodeInfo &node : model.nodes) {
    const OpSpec *spec = findOp(node.opType);
    if (!spec)
      return makeError("unsupported op '" + node.opType +
                       "'; supported: Constant, Add, Relu, MatMul");
    if (node.inputs.size() != spec->numInputs || node.outputs.size() != 1)
      return makeError(node.opType + " expects " +
                       llvm::Twine(spec->numInputs) + " input(s), 1 output");

    SmallVector<Value> operands;
    for (const std::string &name : node.inputs) {
      Expected<Value> operand = lookup(name);
      if (!operand)
        return operand.takeError();
      operands.push_back(*operand);
    }

    Expected<Value> result =
        spec->build(builder, locFor(node.outputs[0]), operands);
    if (!result)
      return result.takeError();
    values[node.outputs[0]] = *result;
  }

  SmallVector<Value> results;
  for (const TensorDesc &output : model.outputs) {
    Expected<Value> value = lookup(output.name);
    if (!value)
      return value.takeError();
    results.push_back(*value);
  }
  func::ReturnOp::create(builder, fileLoc, results);

  // Catches, e.g., a declared output shape that differs from the computed one.
  if (failed(verify(*module)))
    return makeError("generated MLIR failed verification");
  return module;
}
