//===- JIT.cpp - Run a lowered module on the host CPU -------------*- C++ -*-===//
//
// After tc-bufferize, `@main(inputs...) -> results...` has become
// `@main(inputs..., outputs...)` over memrefs, and MLIRGen asked for a C
// wrapper, `_mlir_ciface_main`, that takes each memref as a pointer to a
// descriptor struct. This file builds those descriptors around host buffers
// and calls the wrapper through ExecutionEngine.
//
//===----------------------------------------------------------------------===//

#include "runtime/JIT.h"

#include "support/Common.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::tc;

/// A ranked memref descriptor for a row-major buffer, in MLIR's C layout:
///   { T *allocated; T *aligned; intptr_t offset;
///     intptr_t sizes[rank]; intptr_t strides[rank]; }
/// Every field is pointer-sized, so it is stored as a flat intptr_t array.
static SmallVector<intptr_t> makeDescriptor(float *data,
                                            ArrayRef<int64_t> shape) {
  SmallVector<intptr_t> desc = {reinterpret_cast<intptr_t>(data),
                                reinterpret_cast<intptr_t>(data), 0};
  desc.append(shape.begin(), shape.end());
  SmallVector<intptr_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  desc.append(strides.begin(), strides.end());
  return desc;
}

llvm::Expected<SmallVector<TensorBuffer>>
mlir::tc::runJIT(ModuleOp module, ArrayRef<TensorBuffer> inputs,
                 ArrayRef<std::vector<int64_t>> outputShapes) {
  static const bool nativeTargetReady = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)nativeTargetReady;
  registerBuiltinDialectTranslation(*module.getContext());
  registerLLVMDialectTranslation(*module.getContext());

  SmallVector<TensorBuffer> outputs;
  for (const std::vector<int64_t> &shape : outputShapes)
    outputs.push_back(
        {shape, std::vector<float>(static_cast<size_t>(numElements(shape)))});

  // Argument order after BufferResultsToOutParams: inputs, then outputs.
  // @main only reads its inputs; the C ABI just isn't const-qualified.
  SmallVector<SmallVector<intptr_t>> descriptors;
  for (const TensorBuffer &in : inputs)
    descriptors.push_back(
        makeDescriptor(const_cast<float *>(in.data.data()), in.shape));
  for (TensorBuffer &out : outputs)
    descriptors.push_back(makeDescriptor(out.data.data(), out.shape));

  auto engine = ExecutionEngine::create(module);
  if (!engine)
    return engine.takeError();

  // `_mlir_ciface_main` takes a pointer to each descriptor, and invokePacked
  // takes a pointer to each argument: hence two levels of indirection.
  SmallVector<void *> descriptorPtrs = llvm::map_to_vector(
      descriptors, [](SmallVector<intptr_t> &d) -> void * { return d.data(); });
  SmallVector<void *> args;
  for (void *&ptr : descriptorPtrs)
    args.push_back(&ptr);

  if (llvm::Error err = (*engine)->invokePacked("_mlir_ciface_main", args))
    return std::move(err);
  return outputs;
}
