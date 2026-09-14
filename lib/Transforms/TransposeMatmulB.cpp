//===- TransposeMatmulB.cpp - linalg.matmul -> transpose-B --------*- C++ -*-===//

#include "tc/Conversion/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tc {
#define GEN_PASS_DEF_TRANSPOSEMATMULB
#include "tc/Conversion/Passes.h.inc"
} // namespace tc
} // namespace mlir

using namespace mlir;

namespace {

struct TransposeMatmulBPass
    : public mlir::tc::impl::TransposeMatmulBBase<TransposeMatmulBPass> {
  using TransposeMatmulBBase::TransposeMatmulBBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateTransposeMatmulPatterns(patterns, /*transposeLHS=*/false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
