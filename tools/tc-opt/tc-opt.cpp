//===- tc-opt.cpp - Mini tensor compiler optimizer driver ------*- C++ -*-===//

#include "lowering/Passes.h"
#include "lowering/Pipeline.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tc::registerTCPasses();
  mlir::tc::registerTCPipelines();

  mlir::DialectRegistry registry;
  mlir::tc::registerTCCompilerDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Mini tensor compiler optimizer\n",
                        registry));
}
