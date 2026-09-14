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
  MLIRGPUDialect
  MLIRGPUPipelines
  MLIRGPUToLLVMIRTranslation
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  MLIRIndexToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRNVVMTarget
  MLIRNVVMToLLVMIRTranslation
  MLIRReconcileUnrealizedCasts
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRSCFToControlFlow
  MLIRSCFToGPU
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MLIRUBToLLVM
  )
