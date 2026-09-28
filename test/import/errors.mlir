// RUN: not tc-compile %tc_src/models/invalid_add_shapes.onnx -emit=mlir 2>&1 | FileCheck %s

// A shape error is reported at the ONNX value it would define, then turned
// into a clean tool error (no crash from the inferring builder).
// CHECK: loc("C"({{.*}})): error: operands must have the same shape or one must be a 0-D scalar
// CHECK: tc-compile: invalid operands for tc.add
