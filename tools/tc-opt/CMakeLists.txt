set(LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTC
  MLIRTCConversion
  MLIRTCTransforms
  MLIRTCPipeline
)

add_llvm_executable(tc-opt tc-opt.cpp)

llvm_update_compile_flags(tc-opt)
target_link_libraries(tc-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tc-opt)
