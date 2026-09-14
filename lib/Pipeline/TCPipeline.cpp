//===- TCPipeline.cpp - Progressive lowering + CPU JIT -----------*- C++ -*-===//

#include "tc/Pipeline/TCPipeline.h"

#include "tc/Conversion/Passes.h"
#include "tc/Dialect/TCDialect.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstring>
#include <utility>

using namespace mlir;
using namespace mlir::tc;

void mlir::tc::registerTCCompilerDialects(DialectRegistry &registry) {
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registry.insert<TCDialect>();
  registerAllGPUToLLVMIRTranslations(registry);
}

static void markEmitCInterface(ModuleOp module) {
  module.walk([](func::FuncOp func) {
    if (func.getSymName() == "main")
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(func.getContext()));
  });
}

LogicalResult mlir::tc::runPipeline(ModuleOp module, PipelineStage stage,
                                    ArrayRef<int64_t> tileSizes) {
  MLIRContext *ctx = module.getContext();
  PassManager pm(ctx);
  pm.addPass(createConvertTCToLinalg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createTransposeMatmulB());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());

  // CPU lowering uses SCF tiling to expose cache-friendly loops. NVPTX gets
  // its launch grid from the parallel dimensions of the un-tiled Linalg
  // operations below; applying the CPU tiler here would turn those dimensions
  // into scf.for and leave GPU mapping with no parallel work to map.
  if (stage != PipelineStage::NVPTX) {
    TileAndFuseOptions tileOpts;
    if (tileSizes.empty())
      tileOpts.tileSizes = {32, 32, 32};
    else
      tileOpts.tileSizes.assign(tileSizes.begin(), tileSizes.end());
    pm.addPass(createTileAndFuse(tileOpts));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  if (stage == PipelineStage::Linalg)
    return pm.run(module);

  bufferization::OneShotBufferizePassOptions bufferizeOpts;
  bufferizeOpts.bufferizeFunctionBoundaries = true;
  bufferizeOpts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOpts));

  bufferization::BufferResultsToOutParamsPassOptions outParams;
  outParams.modifyPublicFunctions = true;
  outParams.hoistStaticAllocs = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(outParams));

  bufferization::buildBufferDeallocationPipeline(pm);

  if (stage == PipelineStage::NVPTX) {
    pm.addPass(createConvertLinalgToParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    pm.addPass(createConvertParallelLoopToGpuPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    gpu::GPUToNVVMPipelineOptions nvvmOpts;
    nvvmOpts.cubinFormat = "isa";
    nvvmOpts.cubinChip = "sm_75";
    gpu::buildLowerToNVVMPassPipeline(pm, nvvmOpts);
    return pm.run(module);
  }

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  if (stage == PipelineStage::Memref)
    return pm.run(module);

  markEmitCInterface(module);
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
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
  return pm.run(module);
}

static int64_t numElements(ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

TensorBuffer mlir::tc::makeSequentialInput(ArrayRef<int64_t> shape) {
  TensorBuffer buf;
  buf.shape.assign(shape.begin(), shape.end());
  buf.data.resize(static_cast<size_t>(numElements(shape)));
  for (size_t i = 0, e = buf.data.size(); i < e; ++i)
    buf.data[i] = static_cast<float>(i);
  return buf;
}

void mlir::tc::printTensorBuffer(const TensorBuffer &buffer, raw_ostream &os) {
  os << "shape:";
  if (buffer.shape.empty()) {
    os << " scalar\n";
  } else {
    for (size_t i = 0; i < buffer.shape.size(); ++i) {
      if (i)
        os << "x";
      os << buffer.shape[i];
    }
    os << "\n";
  }
  int64_t inner = buffer.shape.empty() ? 1 : buffer.shape.back();
  for (size_t i = 0, e = buffer.data.size(); i < e; ++i) {
    if (i && inner > 0 && (i % static_cast<size_t>(inner)) == 0)
      os << "\n";
    else if (i)
      os << " ";
    os << buffer.data[i];
  }
  os << "\n";
}

static llvm::Error makeError(const Twine &msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
}

llvm::Expected<TensorBuffer> mlir::tc::loadNpyF32(StringRef path) {
  auto fileOr = llvm::MemoryBuffer::getFile(path);
  if (!fileOr)
    return makeError("failed to read '" + path + "': " +
                     fileOr.getError().message());

  StringRef bytes = fileOr.get()->getBuffer();
  if (!bytes.starts_with("\x93NUMPY"))
    return makeError("'" + path + "' is not a .npy file");
  if (bytes.size() < 10)
    return makeError("truncated .npy header in '" + path + "'");

  uint8_t major = static_cast<uint8_t>(bytes[6]);
  size_t headerLen = 0;
  size_t headerStart = 0;
  if (major == 1) {
    uint16_t len = 0;
    memcpy(&len, bytes.data() + 8, sizeof(len));
    headerLen = len;
    headerStart = 10;
  } else if (major == 2 || major == 3) {
    uint32_t len = 0;
    memcpy(&len, bytes.data() + 8, sizeof(len));
    headerLen = len;
    headerStart = 12;
  } else {
    return makeError("unsupported .npy version in '" + path + "'");
  }

  if (bytes.size() < headerStart + headerLen)
    return makeError("truncated .npy header in '" + path + "'");

  StringRef header = bytes.substr(headerStart, headerLen);
  if (!header.contains("'descr': '<f4'") && !header.contains("\"descr\": \"<f4\"") &&
      !header.contains("'descr': '|f4'"))
    return makeError("'" + path + "' must be little-endian float32");
  if (header.contains("'fortran_order': True") ||
      header.contains("\"fortran_order\": true"))
    return makeError("'" + path + "' must be C-contiguous");

  TensorBuffer buf;
  auto shapePos = header.find("'shape':");
  if (shapePos == StringRef::npos)
    shapePos = header.find("\"shape\":");
  if (shapePos == StringRef::npos)
    return makeError("missing shape in '" + path + "'");

  auto open = header.find('(', shapePos);
  auto close = header.find(')', open);
  if (open == StringRef::npos || close == StringRef::npos)
    return makeError("invalid shape in '" + path + "'");

  StringRef dims = header.slice(open + 1, close).trim();
  if (!dims.empty()) {
    while (!dims.empty()) {
      auto [head, rest] = dims.split(',');
      head = head.trim();
      if (!head.empty()) {
        int64_t dim = 0;
        if (head.getAsInteger(10, dim))
          return makeError("invalid shape dim in '" + path + "'");
        buf.shape.push_back(dim);
      }
      dims = rest.trim();
    }
  }

  StringRef payload = bytes.drop_front(headerStart + headerLen);
  size_t expected = static_cast<size_t>(numElements(buf.shape)) * sizeof(float);
  if (payload.size() < expected)
    return makeError("truncated payload in '" + path + "'");

  buf.data.resize(expected / sizeof(float));
  memcpy(buf.data.data(), payload.data(), expected);
  return buf;
}

namespace {

class HostMemRef {
public:
  explicit HostMemRef(TensorBuffer buffer) : buffer(std::move(buffer)) {}

  void *ptr() { return desc.data(); }
  TensorBuffer takeBuffer() { return std::move(buffer); }

  void pack() {
    size_t rank = buffer.shape.size();
    size_t bytes =
        2 * sizeof(void *) + sizeof(intptr_t) + 2 * rank * sizeof(intptr_t);
    desc.assign(bytes, 0);
    char *p = desc.data();
    float *data = buffer.data.data();
    memcpy(p, &data, sizeof(void *));
    p += sizeof(void *);
    memcpy(p, &data, sizeof(void *));
    p += sizeof(void *);
    intptr_t offset = 0;
    memcpy(p, &offset, sizeof(intptr_t));
    p += sizeof(intptr_t);
    for (int64_t d : buffer.shape) {
      intptr_t size = static_cast<intptr_t>(d);
      memcpy(p, &size, sizeof(intptr_t));
      p += sizeof(intptr_t);
    }
    SmallVector<intptr_t> strides(rank);
    if (rank > 0) {
      strides[rank - 1] = 1;
      for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * static_cast<intptr_t>(buffer.shape[i + 1]);
    }
    for (intptr_t s : strides) {
      memcpy(p, &s, sizeof(intptr_t));
      p += sizeof(intptr_t);
    }
  }

  TensorBuffer buffer;
  std::vector<char> desc;
};

} // namespace

llvm::Error mlir::tc::runJIT(ModuleOp module, ArrayRef<TensorBuffer> inputs,
                             MutableArrayRef<TensorBuffer> outputs) {
  static const bool nativeInit = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)nativeInit;

  for (TensorBuffer &out : outputs) {
    size_t n = static_cast<size_t>(numElements(out.shape));
    if (out.data.size() != n)
      out.data.assign(n, 0.0f);
  }

  SmallVector<HostMemRef> hostArgs;
  hostArgs.reserve(inputs.size() + outputs.size());
  for (const TensorBuffer &in : inputs)
    hostArgs.emplace_back(in);
  for (TensorBuffer &out : outputs)
    hostArgs.emplace_back(std::move(out));
  for (HostMemRef &arg : hostArgs)
    arg.pack();

  auto engineOr = ExecutionEngine::create(module);
  if (!engineOr)
    return engineOr.takeError();

  // `_mlir_ciface_main` takes pointers to memref descriptors. The packed
  // wrapper loads a pointer from each slot, so pass the address of each
  // descriptor pointer rather than the descriptor itself.
  SmallVector<void *> descPtrs;
  descPtrs.reserve(hostArgs.size());
  for (HostMemRef &arg : hostArgs)
    descPtrs.push_back(arg.ptr());
  SmallVector<void *> packed;
  packed.reserve(descPtrs.size());
  for (void *&ptr : descPtrs)
    packed.push_back(&ptr);

  if (llvm::Error err =
          engineOr.get()->invokePacked("_mlir_ciface_main", packed))
    return err;

  for (size_t i = 0, e = outputs.size(); i < e; ++i)
    outputs[i] = hostArgs[inputs.size() + i].takeBuffer();
  return llvm::Error::success();
}

LogicalResult mlir::tc::emitPTX(ModuleOp module, raw_ostream &os) {
  bool found = false;
  module.walk([&](gpu::BinaryOp binary) {
    for (Attribute attr : binary.getObjects()) {
      auto object = dyn_cast<gpu::ObjectAttr>(attr);
      if (!object)
        continue;
      if (object.getFormat() != gpu::CompilationTarget::Assembly)
        continue;
      StringRef ptx = object.getObject().getValue();
      if (ptx.empty())
        continue;
      if (found)
        os << "\n// --- " << binary.getName() << " ---\n";
      os << ptx;
      if (!ptx.ends_with("\n"))
        os << "\n";
      found = true;
    }
  });
  if (!found) {
    module.emitError("no serialized PTX (gpu.binary assembly) in module");
    return failure();
  }
  return success();
}
