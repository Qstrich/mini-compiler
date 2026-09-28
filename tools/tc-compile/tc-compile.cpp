//===- tc-compile.cpp - Mini tensor compiler driver ------------*- C++ -*-===//
//
// ONNX file -> ModelInfo -> tc -> [pipelines] -> print or JIT. `-emit=` picks
// where to stop; see compile() for the whole flow.
//
//===----------------------------------------------------------------------===//

#include "import/MLIRGen.h"
#include "import/ModelInfo.h"
#include "import/ONNXImporter.h"
#include "lowering/Pipeline.h"
#include "runtime/JIT.h"
#include "runtime/TensorBuffer.h"
#include "support/Common.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

using namespace mlir;
using namespace mlir::tc;

namespace {
/// Compilation stages in order; `-emit=X` stops after stage X.
enum class EmitKind { Proto, Mlir, Linalg, Memref, LLVM, JIT };

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
        clEnumValN(EmitKind::JIT, "jit", "JIT-compile and run on CPU")),
    llvm::cl::init(EmitKind::Mlir));

llvm::cl::list<std::string> InputFiles(
    "input", llvm::cl::desc("C-contiguous float32 .npy file for each @main arg"),
    llvm::cl::value_desc("file.npy"), llvm::cl::ZeroOrMore);

llvm::cl::opt<std::string> OutputFilename(
    "o", llvm::cl::desc("Output file (default: stdout)"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));
} // namespace

/// One tensor per graph input: from `-input=` files if given, else 0, 1, 2, ...
static llvm::Expected<SmallVector<TensorBuffer>>
loadInputs(const ModelInfo &model) {
  if (!InputFiles.empty() && InputFiles.size() != model.inputs.size())
    return makeError("expected " + llvm::Twine(model.inputs.size()) +
                     " -input files, got " + llvm::Twine(InputFiles.size()));

  SmallVector<TensorBuffer> inputs;
  for (auto [i, desc] : llvm::enumerate(model.inputs)) {
    if (InputFiles.empty()) {
      inputs.push_back(makeSequentialTensor(desc.shape));
      continue;
    }
    llvm::Expected<TensorBuffer> loaded = loadNpyF32(InputFiles[i]);
    if (!loaded)
      return loaded.takeError();
    if (loaded->shape != desc.shape)
      return makeError("-input #" + llvm::Twine(i) + " shape mismatch");
    inputs.push_back(std::move(*loaded));
  }
  return inputs;
}

/// The whole compiler, one stage per block, stopping where -emit= says.
static llvm::Error compile(llvm::raw_ostream &out) {
  // import, step 1: .onnx file -> ModelInfo
  llvm::Expected<ModelInfo> model = parseONNXFile(InputFilename);
  if (!model)
    return model.takeError();
  if (Emit == EmitKind::Proto) {
    dumpModelInfo(*model, out);
    return llvm::Error::success();
  }

  // import, step 2: ModelInfo -> tc IR
  DialectRegistry registry;
  registerTCCompilerDialects(registry);
  MLIRContext context(registry);
  auto module = generateMLIR(context, *model, InputFilename);
  if (!module)
    return module.takeError();
  if (Emit == EmitKind::Mlir) {
    (*module)->print(out);
    return llvm::Error::success();
  }

  // lowering: tc IR -> Linalg -> memref loops -> LLVM dialect
  PipelineStage stage = Emit == EmitKind::Linalg   ? PipelineStage::Linalg
                        : Emit == EmitKind::Memref ? PipelineStage::Memref
                                                   : PipelineStage::LLVM;
  if (failed(runPipeline(**module, stage)))
    return makeError("lowering failed");
  if (Emit != EmitKind::JIT) {
    (*module)->print(out);
    return llvm::Error::success();
  }

  // runtime: JIT-compile and run @main on this CPU
  auto inputs = loadInputs(*model);
  if (!inputs)
    return inputs.takeError();
  auto outputShapes = llvm::map_to_vector(
      model->outputs, [](const TensorDesc &t) { return t.shape; });
  auto outputs = runJIT(**module, *inputs, outputShapes);
  if (!outputs)
    return makeError("JIT failed: " + llvm::toString(outputs.takeError()));
  for (auto [i, buffer] : llvm::enumerate(*outputs)) {
    out << "output[" << i << "]\n";
    printTensorBuffer(buffer, out);
  }
  return llvm::Error::success();
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "tc-compile - Mini end-to-end tensor compiler (ONNX -> MLIR -> LLVM)\n");

  std::error_code ec;
  llvm::ToolOutputFile output(OutputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "tc-compile: failed to open '" << OutputFilename
                 << "': " << ec.message() << "\n";
    return EXIT_FAILURE;
  }
  if (llvm::Error err = compile(output.os())) {
    llvm::errs() << "tc-compile: " << llvm::toString(std::move(err)) << "\n";
    return EXIT_FAILURE;
  }
  output.keep();
  return EXIT_SUCCESS;
}
