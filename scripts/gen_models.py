#!/usr/bin/env python3
"""Generate static fp32 ONNX sample graphs into models/.

Writes official ONNX protobuf (field numbers from third_party/onnx/onnx.proto)
without requiring the `onnx` Python package.
"""

from __future__ import annotations

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"

# Wire types
VARINT = 0
FIXED32 = 5
LEN = 2

# TensorProto.DataType
FLOAT = 1


def key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _varint(value: int) -> bytes:
    out = bytearray()
    v = value
    while True:
        bits = v & 0x7F
        v >>= 7
        out.append(bits | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _len_field(field: int, payload: bytes) -> bytes:
    return key(field, LEN) + _varint(len(payload)) + payload


def _varint_field(field: int, value: int) -> bytes:
    return key(field, VARINT) + _varint(value)


def _string_field(field: int, value: str) -> bytes:
    data = value.encode("utf-8")
    return _len_field(field, data)


def _float_le(value: float) -> bytes:
    return struct.pack("<f", value)


def encode_shape(dims: list[int]) -> bytes:
    out = bytearray()
    for dim in dims:
        # TensorShapeProto.Dimension.dim_value = 1
        out += _len_field(1, _varint_field(1, dim))
    return bytes(out)


def encode_type(shape: list[int]) -> bytes:
    tensor = _varint_field(1, FLOAT) + _len_field(2, encode_shape(shape))
    # TypeProto.tensor_type = 1
    return _len_field(1, tensor)


def encode_value_info(name: str, shape: list[int]) -> bytes:
    return _string_field(1, name) + _len_field(2, encode_type(shape))


def encode_tensor(name: str, shape: list[int], data: list[float]) -> bytes:
    out = bytearray()
    for dim in shape:
        out += _varint_field(1, dim)
    out += _varint_field(2, FLOAT)
    out += _string_field(8, name)
    raw = b"".join(_float_le(x) for x in data)
    out += _len_field(9, raw)
    return bytes(out)


def encode_node(op_type: str, inputs: list[str], outputs: list[str],
                attrs: list[bytes] | None = None) -> bytes:
    out = bytearray()
    for name in inputs:
        out += _string_field(1, name)
    for name in outputs:
        out += _string_field(2, name)
    out += _string_field(4, op_type)
    for attr in attrs or []:
        out += _len_field(5, attr)
    return bytes(out)


def encode_graph(name: str, nodes: list[bytes], inputs: list[bytes],
                 outputs: list[bytes], initializers: list[bytes]) -> bytes:
    out = bytearray()
    for node in nodes:
        out += _len_field(1, node)
    out += _string_field(2, name)
    for init in initializers:
        out += _len_field(5, init)
    for item in inputs:
        out += _len_field(11, item)
    for item in outputs:
        out += _len_field(12, item)
    return bytes(out)


def encode_opset(version: int = 13) -> bytes:
    # OperatorSetIdProto: empty domain (ai.onnx), version = 2.
    return _varint_field(2, version)


def encode_model(graph: bytes) -> bytes:
    # ir_version = 1, graph = 7, opset_import = 8.
    return (
        _varint_field(1, 8)
        + _len_field(7, graph)
        + _len_field(8, encode_opset(13))
    )


def write_model(path: Path, graph: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_model(graph))
    print(f"wrote {path}")


def gen_add() -> None:
    graph = encode_graph(
        "add",
        [encode_node("Add", ["A", "B"], ["C"])],
        [encode_value_info("A", [2, 2]), encode_value_info("B", [2, 2])],
        [encode_value_info("C", [2, 2])],
        [],
    )
    write_model(OUT / "add.onnx", graph)


def gen_relu() -> None:
    graph = encode_graph(
        "relu",
        [encode_node("Relu", ["X"], ["Y"])],
        [encode_value_info("X", [2, 2])],
        [encode_value_info("Y", [2, 2])],
        [],
    )
    write_model(OUT / "relu.onnx", graph)


def gen_matmul() -> None:
    graph = encode_graph(
        "matmul",
        [encode_node("MatMul", ["A", "B"], ["C"])],
        [encode_value_info("A", [2, 4]), encode_value_info("B", [4, 3])],
        [encode_value_info("C", [2, 3])],
        [],
    )
    write_model(OUT / "matmul.onnx", graph)


def gen_linear() -> None:
    w = [float(i) for i in range(12)]
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    graph = encode_graph(
        "linear",
        [
            encode_node("MatMul", ["X", "W"], ["t1"]),
            encode_node("Add", ["t1", "b"], ["t2"]),
            encode_node("Relu", ["t2"], ["Y"]),
        ],
        [encode_value_info("X", [2, 4])],
        [encode_value_info("Y", [2, 3])],
        [
            encode_tensor("W", [4, 3], w),
            encode_tensor("b", [2, 3], b),
        ],
    )
    write_model(OUT / "linear.onnx", graph)


if __name__ == "__main__":
    gen_add()
    gen_relu()
    gen_matmul()
    gen_linear()
