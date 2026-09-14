//===- TileAndFuse.cpp - Tile matmul and fuse epilogues ----------*- C++ -*-===//

#include "tc/Conversion/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace tc {
#define GEN_PASS_DEF_TILEANDFUSE
#include "tc/Conversion/Passes.h.inc"
} // namespace tc
} // namespace mlir

using namespace mlir;

namespace {

static bool isNestedInLoop(Operation *op) {
  return op->getParentOfType<LoopLikeOpInterface>() != nullptr;
}

static bool isElementwiseGeneric(Operation *op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic || !generic.hasPureTensorSemantics())
    return false;
  return llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType t) {
                        return t == utils::IteratorType::parallel;
                      });
}

static SmallVector<int64_t> effectiveTileSizes(ArrayRef<int64_t> sizes) {
  if (sizes.empty())
    return {32, 32, 32};
  return SmallVector<int64_t>(sizes.begin(), sizes.end());
}

static bool allZero(ArrayRef<int64_t> sizes) {
  return llvm::all_of(sizes, [](int64_t s) { return s == 0; });
}

static SmallVector<OpFoldResult> sizesForOp(MLIRContext *ctx,
                                            ArrayRef<int64_t> tileSizes,
                                            unsigned numLoops) {
  SmallVector<int64_t> trimmed(tileSizes.begin(), tileSizes.end());
  if (trimmed.size() > numLoops)
    trimmed.resize(numLoops);
  while (trimmed.size() < numLoops)
    trimmed.push_back(0);
  return getAsIndexOpFoldResult(ctx, trimmed);
}

// Contractions accumulate into the dest. `tensor.empty` is undef, which is
// fine for a single full matmul that happens to see zeros, but tiled slices
// leak garbage into C[i,j] += ....
static void zeroInitReductionDests(RewriterBase &rewriter, TilingInterface op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
  if (!linalgOp)
    return;
  if (llvm::none_of(op.getLoopIteratorTypes(), [](utils::IteratorType t) {
        return t == utils::IteratorType::reduction;
      }))
    return;

  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  for (OpOperand &init : linalgOp.getDpsInitsMutable()) {
    Value dest = init.get();
    if (dest.getDefiningOp<linalg::FillOp>())
      continue;
    Type elemTy = getElementTypeOrSelf(dest.getType());
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(elemTy));
    Value filled =
        linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{dest})
            .getResult(0);
    init.set(filled);
  }
}

static LogicalResult tileOp(RewriterBase &rewriter, TilingInterface op,
                            ArrayRef<int64_t> tileSizes,
                            SmallVectorImpl<LoopLikeOpInterface> *loopsOut) {
  unsigned numLoops = op.getLoopIteratorTypes().size();
  if (numLoops == 0)
    return success();

  SmallVector<OpFoldResult> ofr = sizesForOp(op.getContext(), tileSizes, numLoops);
  // Keep reduction dims untiled. Tiling K requires partial-reduction init
  // handling; leaving K inside the named matmul is numerically safe and still
  // produces M/N `scf.for` tiles.
  for (auto [i, iter] : llvm::enumerate(op.getLoopIteratorTypes())) {
    if (iter == utils::IteratorType::reduction)
      ofr[i] = rewriter.getIndexAttr(0);
  }
  if (allZero(tileSizes) ||
      llvm::all_of(ofr, [](OpFoldResult f) {
        auto v = getConstantIntValue(f);
        return v && *v == 0;
      }))
    return success();

  scf::SCFTilingOptions options;
  options.setTileSizes(ofr);

  zeroInitReductionDests(rewriter, op);
  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFTilingResult> tiled =
      scf::tileUsingSCF(rewriter, op, options);
  if (failed(tiled))
    return failure();

  rewriter.replaceOp(op, tiled->replacements);
  if (loopsOut)
    *loopsOut = tiled->loops;
  return success();
}

static void fuseElementwiseConsumers(RewriterBase &rewriter,
                                     MutableArrayRef<LoopLikeOpInterface> loops) {
  if (loops.empty() || loops.front()->getNumResults() == 0)
    return;

  llvm::SmallPtrSet<Operation *, 8> fusedOps;
  Value current = loops.front()->getResult(0);
  while (current && current.hasOneUse()) {
    Operation *user = *current.getUsers().begin();
    if (!isElementwiseGeneric(user) || fusedOps.contains(user))
      break;

    rewriter.setInsertionPoint(user);
    FailureOr<scf::SCFFuseConsumerOfSliceResult> fused =
        scf::tileAndFuseConsumer(rewriter, user, loops);
    if (failed(fused) || fused->tiledOps.empty())
      break;

    fusedOps.insert(user);
    // tileAndFuseConsumer rewrites uses of the consumer results to the new
    // loop yields but leaves the original op in the IR. Erase it so we do not
    // try to fuse the same consumer forever.
    if (user->use_empty())
      rewriter.eraseOp(user);

    Operation *outer = loops.front();
    if (outer->getNumResults() == 0)
      break;
    current = outer->getResult(outer->getNumResults() - 1);
  }
}

static LogicalResult runOnFunc(func::FuncOp func, ArrayRef<int64_t> tileSizes) {
  IRRewriter rewriter(func.getContext());

  SmallVector<linalg::MatmulOp> matmuls;
  func.walk([&](linalg::MatmulOp op) {
    if (!isNestedInLoop(op) && op.hasPureTensorSemantics())
      matmuls.push_back(op);
  });

  for (linalg::MatmulOp op : matmuls) {
    if (!op->getBlock())
      continue;
    SmallVector<LoopLikeOpInterface> loops;
    if (failed(tileOp(rewriter, cast<TilingInterface>(op.getOperation()),
                      tileSizes, &loops)))
      return failure();
    if (loops.empty())
      continue;
    fuseElementwiseConsumers(rewriter, loops);
  }

  SmallVector<linalg::GenericOp> generics;
  func.walk([&](linalg::GenericOp op) {
    if (!isNestedInLoop(op) && isElementwiseGeneric(op))
      generics.push_back(op);
  });

  for (linalg::GenericOp op : generics) {
    if (!op->getBlock())
      continue;
    if (failed(tileOp(rewriter, cast<TilingInterface>(op.getOperation()),
                      tileSizes, /*loopsOut=*/nullptr)))
      return failure();
  }
  return success();
}

struct TileAndFusePass : public mlir::tc::impl::TileAndFuseBase<TileAndFusePass> {
  using TileAndFuseBase::TileAndFuseBase;

  void runOnOperation() override {
    SmallVector<int64_t> sizes = effectiveTileSizes(tileSizes);
    if (allZero(sizes))
      return;

    WalkResult result = getOperation()->walk([&](func::FuncOp func) {
      if (failed(runOnFunc(func, sizes)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
