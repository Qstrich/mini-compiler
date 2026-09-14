// RUN: tc-compile %tc_src/models/add.onnx -emit=nvptx | FileCheck %s --check-prefix=ADD
// RUN: tc-compile %tc_src/models/linear.onnx -emit=nvptx | FileCheck %s --check-prefix=LINEAR
// RUN: tc-compile %tc_src/models/matmul.onnx -emit=nvptx | FileCheck %s --check-prefix=MATMUL
// RUN: tc-compile %tc_src/models/linear.onnx -emit=nvptx -o %t.ptx && FileCheck %s --check-prefix=LINEAR < %t.ptx

// ADD: .target sm_75
// ADD: .visible .entry
// ADD: add.rn.f32

// LINEAR: .target sm_75
// LINEAR: .visible .entry
// LINEAR: .maxntid 1, 1, 1
// LINEAR: .reg .b32
// LINEAR: mov.u32 {{.*}}, %ctaid.y;
// LINEAR: mov.u32 {{.*}}, %ctaid.x;
// LINEAR: mul.rn.f32
// LINEAR: max.f32

// MATMUL: .target sm_75
// MATMUL: .visible .entry
// MATMUL: mov.u32 {{.*}}, %ctaid.y;
// MATMUL: mov.u32 {{.*}}, %ctaid.x;
// MATMUL: mul.rn.f32
