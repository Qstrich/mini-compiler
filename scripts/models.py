"""The sample models, one spec each: graph, weights, and NumPy reference.

gen_models.py turns these specs into models/<name>.onnx; check_numeric.py
compares tc-compile's JIT output against `reference`. Specs are plain NumPy
data so the numeric check (run by `check-tc`) does not need the onnx package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Shape = tuple[int, ...]
Values = dict[str, np.ndarray]


@dataclass(frozen=True)
class Node:
    op: str
    inputs: list[str]
    output: str


@dataclass(frozen=True)
class Model:
    name: str
    inputs: dict[str, Shape]  # graph inputs = @main arguments, in order
    outputs: dict[str, Shape]
    nodes: list[Node]
    # NumPy result, given every graph input and initializer by name.
    reference: Callable[[Values], np.ndarray] | None = None
    initializers: Values = field(default_factory=dict)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


MODELS: list[Model] = [
    Model(
        name="add",
        inputs={"A": (2, 2), "B": (2, 2)},
        outputs={"C": (2, 2)},
        nodes=[Node("Add", ["A", "B"], "C")],
        reference=lambda v: v["A"] + v["B"],
    ),
    Model(
        name="relu",
        inputs={"X": (2, 2)},
        outputs={"Y": (2, 2)},
        nodes=[Node("Relu", ["X"], "Y")],
        reference=lambda v: relu(v["X"]),
    ),
    Model(
        name="matmul",
        inputs={"A": (2, 4), "B": (4, 3)},
        outputs={"C": (2, 3)},
        nodes=[Node("MatMul", ["A", "B"], "C")],
        reference=lambda v: v["A"] @ v["B"],
    ),
    Model(
        name="linear",
        inputs={"X": (2, 4)},
        outputs={"Y": (2, 3)},
        nodes=[
            Node("MatMul", ["X", "W"], "t1"),
            Node("Add", ["t1", "b"], "t2"),
            Node("Relu", ["t2"], "Y"),
        ],
        initializers={
            "W": np.arange(12, dtype=np.float32).reshape(4, 3),
            "b": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        },
        reference=lambda v: relu(v["X"] @ v["W"] + v["b"]),
    ),
]

# Graphs tc-compile must reject with a clean error (test/Frontend/errors.mlir).
INVALID_MODELS: list[Model] = [
    Model(
        name="invalid_add_shapes",
        inputs={"A": (2, 2), "B": (3, 3)},
        outputs={"C": (2, 2)},
        nodes=[Node("Add", ["A", "B"], "C")],
    ),
]
