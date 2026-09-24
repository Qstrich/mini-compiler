add_mlir_library(MLIRTCPipeline
  TCPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tc

  DEPENDS
  MLIRTCPassesIncGen

  LINK_LIBS PUBLIC
  MLIRTC
  MLIRTCConversion
  MLIRTCTransforms
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
  MLIRRegisterAllExtensions
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MLIRUBToLLVM
  )
