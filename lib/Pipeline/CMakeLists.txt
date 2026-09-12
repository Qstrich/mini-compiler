add_mlir_library(MLIRTCPipeline
  TCPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tc

  DEPENDS
  MLIRTCPassesIncGen

  LINK_LIBS PUBLIC
  MLIRTC
  MLIRTCConversion
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowToLLVM
  MLIRExecutionEngine
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRReconcileUnrealizedCasts
  MLIRRegisterAllDialects
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRExport
  MLIRUBToLLVM
  )
