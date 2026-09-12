configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TC_TEST_DEPENDS
  FileCheck count not
  tc-opt
  tc-compile
)

add_lit_testsuite(check-tc "Running the mini-compiler regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TC_TEST_DEPENDS}
)
set_target_properties(check-tc PROPERTIES FOLDER "Tests")

add_lit_testsuites(TC ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TC_TEST_DEPENDS})
