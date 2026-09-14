add_mlir_library(MLIRTCTransforms
  TransposeMatmulB.cpp
  TileAndFuse.cpp
  OutlineGPUKernel.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tc

  DEPENDS
  MLIRTCPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransformUtils
  )
