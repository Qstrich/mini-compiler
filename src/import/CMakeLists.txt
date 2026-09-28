# ONNX file -> ModelInfo -> tc IR.
add_mlir_library(TCImport
  OnnxProto.cpp
  ONNXImporter.cpp
  ModelInfo.cpp
  MLIRGen.cpp

  DEPENDS
  MLIRTCOpsIncGen

  LINK_LIBS PUBLIC
  TCDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRSupport
  )
