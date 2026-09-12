add_library(MLIRTCFrontend
  ModelInfo.cpp
  ONNXParser.cpp
  MLIRGen.cpp
)
set_target_properties(MLIRTCFrontend PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(MLIRTCFrontend PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(MLIRTCFrontend
  PUBLIC
    MLIRTC
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
)
