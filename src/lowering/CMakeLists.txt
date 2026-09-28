# tc IR -> LLVM dialect: the convert-tc-to-linalg pass plus the three named
# pipelines built on it.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TC)
add_public_tablegen_target(MLIRTCPassesIncGen)

add_mlir_library(TCLowering
  TCToLinalg.cpp
  Pipeline.cpp

  DEPENDS
  MLIRTCOpsIncGen
  MLIRTCPassesIncGen

  LINK_LIBS PUBLIC
  TCDialect
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRSCFToControlFlow
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRUBToLLVM
  )
