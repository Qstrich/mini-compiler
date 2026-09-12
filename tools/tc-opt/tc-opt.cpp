//===- tc-opt.cpp - Mini tensor compiler optimizer driver ------*- C++ -*-===//

#include "tc/Conversion/Passes.h"
#include "tc/Pipeline/TCPipeline.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tc::registerTCPasses();

  mlir::DialectRegistry registry;
  mlir::tc::registerTCCompilerDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Mini tensor compiler optimizer\n",
                        registry));
}
