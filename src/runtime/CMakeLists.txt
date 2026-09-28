# Run a lowered module on the host CPU.
add_mlir_library(TCRuntime
  TensorBuffer.cpp
  JIT.cpp

  LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native

  LINK_LIBS PUBLIC
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMToLLVMIRTranslation
  )
