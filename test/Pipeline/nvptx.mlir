// RUN: tc-compile %tc_src/models/add.onnx -emit=nvptx | FileCheck %s --check-prefix=ADD
// RUN: tc-compile %tc_src/models/linear.onnx -emit=nvptx | FileCheck %s --check-prefix=LINEAR
// RUN: tc-compile %tc_src/models/linear.onnx -emit=nvptx -o %t.ptx && FileCheck %s --check-prefix=LINEAR < %t.ptx

// ADD: .target sm_75
// ADD: .visible .entry
// ADD: add.rn.f32

// LINEAR: .target sm_75
// LINEAR: .visible .entry
// LINEAR: mul.rn.f32
// LINEAR: max.f32
