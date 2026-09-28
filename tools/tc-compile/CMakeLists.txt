set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
)

set(LIBS
  MLIRTCFrontend
  MLIRTC
  MLIRTCConversion
  MLIRTCPipeline
  MLIRIR
  LLVMSupport
)

add_llvm_executable(tc-compile tc-compile.cpp)

llvm_update_compile_flags(tc-compile)
target_link_libraries(tc-compile PRIVATE ${LIBS})

mlir_check_all_link_libraries(tc-compile)
