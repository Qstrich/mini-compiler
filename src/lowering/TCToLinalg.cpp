//===- TCToLinalg.cpp - Convert tc dialect to linalg -------------*- C++ -*-===//

#include "lowering/Passes.h"
#include "dialect/TCDialect.h"
#include "dialect/TCOps.h"

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
#include "lowering/Passes.h.inc"
} // namespace tc
} // namespace mlir

using namespace mlir;
using namespace mlir::tc;

namespace {

static Value createEmpty(OpBuilder &builder, Location loc, RankedTensorType type) {
  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType());
}

/// Build a `linalg.generic` that applies `body` elementwise to `operands`,
/// writing a fresh tensor of `resultType`. A 0-D operand is broadcast.
static Value
buildElementwise(OpBuilder &builder, Location loc, RankedTensorType resultType,
                 ValueRange operands,
                 function_ref<Value(OpBuilder &, Location, ValueRange)> body) {
  MLIRContext *ctx = builder.getContext();
  int64_t rank = resultType.getRank();
  AffineMap identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
  AffineMap broadcast =
      AffineMap::get(rank, /*symbolCount=*/0, /*results=*/{}, ctx);

  SmallVector<AffineMap> maps;
  for (Value operand : operands)
    maps.push_back(cast<RankedTensorType>(operand.getType()).getRank() == 0
                       ? broadcast
                       : identity);
  maps.push_back(identity); // result
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);

  auto generic = linalg::GenericOp::create(
      builder, loc, TypeRange{resultType}, operands,
      ValueRange{createEmpty(builder, loc, resultType)}, maps, iterators,
      [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
        linalg::YieldOp::create(b, bodyLoc, body(b, bodyLoc, args));
      });
  return generic.getResult(0);
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
    Value sum = buildElementwise(
        rewriter, op.getLoc(), op.getType(), adaptor.getOperands(),
        [](OpBuilder &b, Location loc, ValueRange args) -> Value {
          return arith::AddFOp::create(b, loc, args[0], args[1]);
        });
    rewriter.replaceOp(op, sum);
    return success();
  }
};

struct ConvertRelu : public OpConversionPattern<ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value relu = buildElementwise(
        rewriter, op.getLoc(), op.getType(), adaptor.getOperands(),
        [](OpBuilder &b, Location loc, ValueRange args) -> Value {
          Value zero = arith::ConstantOp::create(
              b, loc, b.getZeroAttr(args[0].getType()));
          return arith::MaximumFOp::create(b, loc, args[0], zero);
        });
    rewriter.replaceOp(op, relu);
    return success();
  }
};

struct ConvertMatMul : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resultType = op.getType();
    // linalg.matmul accumulates into its init (C += A * B), so start from 0.
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value init =
        linalg::FillOp::create(rewriter, loc, zero,
                               createEmpty(rewriter, loc, resultType))
            .getResult(0);
    auto matmul = linalg::MatmulOp::create(
        rewriter, loc, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{init});
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
