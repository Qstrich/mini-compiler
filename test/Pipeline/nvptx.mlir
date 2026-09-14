// RUN: tc-compile %tc_src/models/add.onnx -emit=nvptx | FileCheck %s --check-prefix=ADD
// RUN: tc-compile %tc_src/models/linear.onnx -emit=nvptx | FileCheck %s --check-prefix=LINEAR
// RUN: tc-compile %tc_src/models/matmul.onnx -emit=nvptx | FileCheck %s --check-prefix=MATMUL
// RUN: tc-compile %tc_src/models/linear.onnx -emit=nvptx -tile-sizes=2,2 | FileCheck %s --check-prefix=LINEAR-TILED
// RUN: tc-compile %tc_src/models/linear.onnx -emit=nvptx -o %t.ptx && FileCheck %s --check-prefix=LINEAR < %t.ptx

// ADD: .target sm_75
// ADD: .visible .entry
// ADD: add.rn.f32

// LINEAR: .target sm_75
// LINEAR: .visible .entry
// LINEAR: .maxntid 32, 32, 1
// LINEAR: .reg .b32
// LINEAR: mov.u32 {{.*}}, %tid.x;
// LINEAR: mov.u32 {{.*}}, %tid.y;
// LINEAR: mul.rn.f32
// LINEAR: max.f32

// MATMUL: .target sm_75
// MATMUL: .visible .entry
// MATMUL: .maxntid 32, 32, 1
// MATMUL: mov.u32 {{.*}}, %tid.x;
// MATMUL: mov.u32 {{.*}}, %tid.y;
// MATMUL: mul.rn.f32

// LINEAR-TILED: .target sm_75
// LINEAR-TILED: .visible .entry
// LINEAR-TILED: .maxntid 2, 2, 1
// LINEAR-TILED: mov.u32 {{.*}}, %ctaid.x;
// LINEAR-TILED: mov.u32 {{.*}}, %tid.x;
// LINEAR-TILED: mov.u32 {{.*}}, %tid.y;
