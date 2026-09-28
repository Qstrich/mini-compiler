#!/usr/bin/env python3
"""Write every spec in models.py to models/<name>.onnx."""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper

from models import INVALID_MODELS, MODELS, Model, Shape

OUT = Path(__file__).resolve().parents[1] / "models"


def value_info(name: str, shape: Shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def to_onnx(model: Model) -> onnx.ModelProto:
    graph = helper.make_graph(
        name=model.name,
        nodes=[helper.make_node(n.op, n.inputs, [n.output]) for n in model.nodes],
        inputs=[value_info(n, s) for n, s in model.inputs.items()],
        outputs=[value_info(n, s) for n, s in model.outputs.items()],
        initializer=[numpy_helper.from_array(a, n) for n, a in model.initializers.items()],
    )
    # Pin the IR version: newer onnx releases default to IRs that the installed
    # ONNX Runtime may not load yet.
    return helper.make_model(graph, ir_version=8,
                             opset_imports=[helper.make_opsetid("", 13)])


def write(proto: onnx.ModelProto, name: str) -> None:
    path = OUT / f"{name}.onnx"
    onnx.save(proto, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        proto = to_onnx(model)
        onnx.checker.check_model(proto, full_check=True)
        write(proto, model.name)
    for model in INVALID_MODELS:  # deliberately fails the checker
        write(to_onnx(model), model.name)
