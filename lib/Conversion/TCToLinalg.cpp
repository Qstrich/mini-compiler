//===- TCToLinalg.cpp - Convert tc dialect to linalg -------------*- C++ -*-===//

#include "tc/Conversion/Passes.h"
#include "tc/Dialect/TCDialect.h"
#include "tc/Dialect/TCOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace tc {
#define GEN_PASS_DEF_CONVERTTCTOLINALG
#include "tc/Conversion/Passes.h.inc"
} // namespace tc
} // namespace mlir

using namespace mlir;
using namespace mlir::tc;

namespace {

static Value createEmpty(OpBuilder &builder, Location loc, RankedTensorType type) {
  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType());
}

static AffineMap broadcastOrIdentity(MLIRContext *ctx, int64_t resultRank,
                                     RankedTensorType operandType) {
  if (operandType.getRank() == 0)
    return AffineMap::get(resultRank, /*symbolCount=*/0, /*results=*/{}, ctx);
  return AffineMap::getMultiDimIdentityMap(resultRank, ctx);
}

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto lhsType = cast<RankedTensorType>(adaptor.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(adaptor.getRhs().getType());
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    int64_t rank = resultType.getRank();

    Value empty = createEmpty(rewriter, loc, resultType);
    SmallVector<AffineMap> maps = {
        broadcastOrIdentity(ctx, rank, lhsType),
        broadcastOrIdentity(ctx, rank, rhsType),
        AffineMap::getMultiDimIdentityMap(rank, ctx),
    };
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{empty}, maps,
        iterators, [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
          Value sum = arith::AddFOp::create(b, bodyLoc, args[0], args[1]);
          linalg::YieldOp::create(b, bodyLoc, sum);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct ConvertRelu : public OpConversionPattern<ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    int64_t rank = resultType.getRank();

    Value empty = createEmpty(rewriter, loc, resultType);
    AffineMap identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<AffineMap> maps = {identity, identity};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{empty}, maps, iterators,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
          Value zero = arith::ConstantOp::create(b, bodyLoc,
                                                 b.getZeroAttr(args[0].getType()));
          Value relu = arith::MaximumFOp::create(b, bodyLoc, args[0], zero);
          linalg::YieldOp::create(b, bodyLoc, relu);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct ConvertMatMul : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value empty = createEmpty(rewriter, op.getLoc(), resultType);
    auto matmul = linalg::MatmulOp::create(
        rewriter, op.getLoc(), TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{empty});
    rewriter.replaceOp(op, matmul.getResults());
    return success();
  }
};

struct ConvertTCToLinalgPass
    : public mlir::tc::impl::ConvertTCToLinalgBase<ConvertTCToLinalgPass> {
  using ConvertTCToLinalgBase::ConvertTCToLinalgBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addIllegalDialect<TCDialect>();
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertConstant, ConvertAdd, ConvertRelu, ConvertMatMul>(ctx);

    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
