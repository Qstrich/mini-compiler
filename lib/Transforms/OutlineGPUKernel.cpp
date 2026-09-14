//===- OutlineGPUKernel.cpp - Wrap @main in gpu.launch -----------*- C++ -*-===//

#include "tc/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tc {
#define GEN_PASS_DEF_OUTLINEGPUKERNEL
#include "tc/Conversion/Passes.h.inc"
} // namespace tc
} // namespace mlir

using namespace mlir;

namespace {

struct OutlineGPUKernelPass
    : public mlir::tc::impl::OutlineGPUKernelBase<OutlineGPUKernelPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module.walk([](gpu::LaunchOp) { return WalkResult::interrupt(); })
            .wasInterrupted())
      return;

    auto main = module.lookupSymbol<func::FuncOp>("main");
    if (!main || main.getBody().empty()) {
      module.emitError("tc-outline-gpu-kernel: missing func.func @main");
      signalPassFailure();
      return;
    }

    Block &block = main.getBody().front();
    Operation *term = block.getTerminator();
    if (!isa<func::ReturnOp>(term)) {
      main.emitError("tc-outline-gpu-kernel: @main must return");
      signalPassFailure();
      return;
    }

    OpBuilder builder(&block, block.begin());
    Location loc = main.getLoc();
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
    auto launch = gpu::LaunchOp::create(builder, loc, c1, c1, c1, c1, c1, c1);

    Block &launchBody = launch.getBody().front();
    builder.setInsertionPointToEnd(&launchBody);
    gpu::TerminatorOp::create(builder, loc);

    SmallVector<Operation *> toMove;
    for (Operation &op : block) {
      if (&op == c1.getDefiningOp() || &op == launch.getOperation() ||
          &op == term)
        continue;
      toMove.push_back(&op);
    }
    Operation *launchTerm = launchBody.getTerminator();
    for (Operation *op : toMove)
      op->moveBefore(launchTerm);
  }
};

} // namespace
