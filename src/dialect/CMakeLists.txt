# The tc IR. TableGen writes TCOps*.inc next to the build copy of this folder.
add_mlir_dialect(TCOps tc)
add_mlir_doc(TCDialect TCDialect tc/ -gen-dialect-doc)
add_mlir_doc(TCOps TCOps tc/ -gen-op-doc)

add_mlir_dialect_library(TCDialect
  TCDialect.cpp
  TCOps.cpp

  DEPENDS
  MLIRTCOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
