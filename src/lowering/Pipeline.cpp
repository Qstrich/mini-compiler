//===- Pipeline.cpp - tc -> LLVM lowering pipelines ---------------*- C++ -*-===//

#include "lowering/Pipeline.h"

#include "lowering/Passes.h"
#include "dialect/TCDialect.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void tc::registerTCCompilerDialects(DialectRegistry &registry) {
  registerAllDialects(registry);
  // Bufferization and to-LLVM interfaces for upstream dialects.
  registerAllExtensions(registry);
  registry.insert<TCDialect>();
}

static void addCleanup(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void tc::buildLowerToLinalg(OpPassManager &pm) {
  pm.addPass(createConvertTCToLinalg());
  addCleanup(pm);
}

void tc::buildBufferize(OpPassManager &pm) {
  // Function boundaries use identity-layout memrefs, so the host can pass
  // plain row-major buffers.
  bufferization::OneShotBufferizePassOptions bufferize;
  bufferize.bufferizeFunctionBoundaries = true;
  bufferize.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferize));

  // Returned buffers become caller-allocated out-params (see Runtime/JIT.cpp).
  bufferization::BufferResultsToOutParamsPassOptions outParams;
  outParams.modifyPublicFunctions = true;
  outParams.hoistStaticAllocs = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(outParams));
  bufferization::buildBufferDeallocationPipeline(pm);

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  addCleanup(pm);
}

void tc::buildLowerToLLVM(OpPassManager &pm) {
  pm.addPass(createSCFToControlFlowPass());
  addCleanup(pm);
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createUBToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void tc::registerTCPipelines() {
  PassPipelineRegistration<>("tc-lower-to-linalg",
                             "Lower tc ops to Linalg on tensors",
                             buildLowerToLinalg);
  PassPipelineRegistration<>("tc-bufferize",
                             "Bufferize and lower Linalg to loops",
                             buildBufferize);
  PassPipelineRegistration<>("tc-lower-to-llvm",
                             "Lower loops and memrefs to the LLVM dialect",
                             buildLowerToLLVM);
}

LogicalResult tc::runPipeline(ModuleOp module, PipelineStage stage) {
  PassManager pm(module.getContext());
  buildLowerToLinalg(pm);
  if (stage >= PipelineStage::Memref)
    buildBufferize(pm);
  if (stage >= PipelineStage::LLVM)
    buildLowerToLLVM(pm);
  return pm.run(module);
}
