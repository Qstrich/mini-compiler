//===- tc-compile.cpp - Mini tensor compiler driver ------------*- C++ -*-===//

#include "tc/Frontend/MLIRGen.h"
#include "tc/Frontend/ModelInfo.h"
#include "tc/Frontend/ONNXParser.h"
#include "tc/Pipeline/TCPipeline.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <string>

namespace {
enum class EmitKind {
  Proto,
  Mlir,
  Linalg,
  Memref,
  LLVM,
  JIT,
  NVPTX,
};

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"),
                                         llvm::cl::Required);

llvm::cl::opt<EmitKind> Emit(
    "emit", llvm::cl::desc("Emission stage"),
    llvm::cl::values(
        clEnumValN(EmitKind::Proto, "proto", "Dump structured ONNX ModelInfo"),
        clEnumValN(EmitKind::Mlir, "mlir", "Emit tc dialect MLIR"),
        clEnumValN(EmitKind::Linalg, "linalg", "Emit after linalg lowering"),
        clEnumValN(EmitKind::Memref, "memref", "Emit after bufferization"),
        clEnumValN(EmitKind::LLVM, "llvm", "Emit host LLVM dialect"),
        clEnumValN(EmitKind::JIT, "jit", "JIT-compile and run on CPU"),
        clEnumValN(EmitKind::NVPTX, "nvptx", "Emit NVPTX/PTX (no launch)")),
    llvm::cl::init(EmitKind::Mlir));

llvm::cl::list<std::string> InputData(
    "input", llvm::cl::desc("C-contiguous float32 .npy file for each @main arg"),
    llvm::cl::value_desc("file.npy"), llvm::cl::ZeroOrMore);

llvm::cl::opt<std::string> TileSizes(
    "tile-sizes",
    llvm::cl::desc("Comma-separated M,N,K Linalg tile sizes (default 32,32,32)"),
    llvm::cl::init("32,32,32"));

llvm::cl::opt<std::string> OutputFilename(
    "o", llvm::cl::desc("Output file (default: stdout)"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));
} // namespace

static llvm::SmallVector<int64_t> parseTileSizes(llvm::StringRef text) {
  llvm::SmallVector<int64_t> sizes;
  if (text.empty())
    return {32, 32, 32};
  while (!text.empty()) {
    auto [head, rest] = text.split(',');
    int64_t value = 0;
    if (head.trim().getAsInteger(10, value))
      return {};
    sizes.push_back(value);
    text = rest;
  }
  return sizes;
}

static const char *emitKindName(EmitKind kind) {
  switch (kind) {
  case EmitKind::Proto:
    return "proto";
  case EmitKind::Mlir:
    return "mlir";
  case EmitKind::Linalg:
    return "linalg";
  case EmitKind::Memref:
    return "memref";
  case EmitKind::LLVM:
    return "llvm";
  case EmitKind::JIT:
    return "jit";
  case EmitKind::NVPTX:
    return "nvptx";
  }
  return "unknown";
}

static mlir::tc::PipelineStage emitToStage(EmitKind kind) {
  switch (kind) {
  case EmitKind::Linalg:
    return mlir::tc::PipelineStage::Linalg;
  case EmitKind::Memref:
    return mlir::tc::PipelineStage::Memref;
  case EmitKind::LLVM:
  case EmitKind::JIT:
    return mlir::tc::PipelineStage::LLVM;
  case EmitKind::NVPTX:
    return mlir::tc::PipelineStage::NVPTX;
  default:
    llvm_unreachable("emit kind is not a lowering stage");
  }
}

static llvm::Expected<std::unique_ptr<llvm::ToolOutputFile>>
openOutput() {
  if (OutputFilename == "-")
    return std::unique_ptr<llvm::ToolOutputFile>();
  std::error_code ec;
  auto file = std::make_unique<llvm::ToolOutputFile>(OutputFilename, ec,
                                                     llvm::sys::fs::OF_None);
  if (ec)
    return llvm::createStringError(ec, "failed to open '" + OutputFilename +
                                           "': " + ec.message());
  return file;
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "tc-compile - Mini end-to-end tensor compiler (ONNX -> MLIR -> LLVM)\n");

  llvm::Expected<std::unique_ptr<llvm::ToolOutputFile>> outFileOr =
      openOutput();
  if (!outFileOr) {
    llvm::errs() << "tc-compile: " << llvm::toString(outFileOr.takeError())
                 << "\n";
    return EXIT_FAILURE;
  }
  llvm::raw_ostream &out =
      *outFileOr ? (*outFileOr)->os() : llvm::outs();

  llvm::Expected<mlir::tc::ModelInfo> modelOr =
      mlir::tc::parseONNXFile(InputFilename);
  if (!modelOr) {
    llvm::errs() << "tc-compile: " << llvm::toString(modelOr.takeError())
                 << "\n";
    return EXIT_FAILURE;
  }

  if (Emit == EmitKind::Proto) {
    mlir::tc::dumpModelInfo(*modelOr, out);
    if (*outFileOr)
      (*outFileOr)->keep();
    return EXIT_SUCCESS;
  }

  mlir::DialectRegistry registry;
  mlir::tc::registerTCCompilerDialects(registry);
  mlir::MLIRContext context(registry);

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> moduleOr =
      mlir::tc::generateMLIR(context, *modelOr, InputFilename);
  if (!moduleOr) {
    llvm::errs() << "tc-compile: " << llvm::toString(moduleOr.takeError())
                 << "\n";
    return EXIT_FAILURE;
  }

  if (Emit == EmitKind::Mlir) {
    (*moduleOr)->print(out);
    if (*outFileOr)
      (*outFileOr)->keep();
    return EXIT_SUCCESS;
  }

  llvm::SmallVector<int64_t> tileSizes = parseTileSizes(TileSizes);
  if (tileSizes.empty()) {
    llvm::errs() << "tc-compile: invalid -tile-sizes '" << TileSizes << "'\n";
    return EXIT_FAILURE;
  }

  if (failed(mlir::tc::runPipeline(**moduleOr, emitToStage(Emit), tileSizes))) {
    llvm::errs() << "tc-compile: lowering to " << emitKindName(Emit)
                 << " failed\n";
    return EXIT_FAILURE;
  }

  if (Emit == EmitKind::NVPTX) {
    if (failed(mlir::tc::emitPTX(**moduleOr, out))) {
      llvm::errs() << "tc-compile: failed to extract PTX\n";
      return EXIT_FAILURE;
    }
    if (*outFileOr)
      (*outFileOr)->keep();
    return EXIT_SUCCESS;
  }

  if (Emit != EmitKind::JIT) {
    (*moduleOr)->print(out);
    if (*outFileOr)
      (*outFileOr)->keep();
    return EXIT_SUCCESS;
  }

  if (!InputData.empty() && InputData.size() != modelOr->inputs.size()) {
    llvm::errs() << "tc-compile: expected " << modelOr->inputs.size()
                 << " -input files, got " << InputData.size() << "\n";
    return EXIT_FAILURE;
  }

  llvm::SmallVector<mlir::tc::TensorBuffer> inputs;
  inputs.reserve(modelOr->inputs.size());
  for (size_t i = 0; i < modelOr->inputs.size(); ++i) {
    if (InputData.empty()) {
      inputs.push_back(mlir::tc::makeSequentialInput(modelOr->inputs[i].shape));
      continue;
    }
    llvm::Expected<mlir::tc::TensorBuffer> loaded =
        mlir::tc::loadNpyF32(InputData[i]);
    if (!loaded) {
      llvm::errs() << "tc-compile: " << llvm::toString(loaded.takeError())
                   << "\n";
      return EXIT_FAILURE;
    }
    if (loaded->shape != modelOr->inputs[i].shape) {
      llvm::errs() << "tc-compile: -input #" << i << " shape mismatch\n";
      return EXIT_FAILURE;
    }
    inputs.push_back(std::move(*loaded));
  }

  llvm::SmallVector<mlir::tc::TensorBuffer> outputs;
  outputs.reserve(modelOr->outputs.size());
  for (const mlir::tc::TensorDesc &out : modelOr->outputs) {
    mlir::tc::TensorBuffer buf;
    buf.shape = out.shape;
    buf.data.assign(1, 0.0f);
    size_t n = 1;
    for (int64_t d : out.shape)
      n *= static_cast<size_t>(d);
    buf.data.assign(n, 0.0f);
    outputs.push_back(std::move(buf));
  }

  if (llvm::Error err =
          mlir::tc::runJIT(**moduleOr, inputs, outputs)) {
    llvm::errs() << "tc-compile: JIT failed: " << llvm::toString(std::move(err))
                 << "\n";
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    llvm::outs() << "output[" << i << "]\n";
    mlir::tc::printTensorBuffer(outputs[i], llvm::outs());
  }
  return EXIT_SUCCESS;
}
