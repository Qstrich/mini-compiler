#!/usr/bin/env python3
"""Compare tc-compile -emit=jit output to a NumPy reference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REL_TOL = 1e-4
ABS_TOL = 1e-5


def sequential(shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape)) if shape else 1
    return np.arange(n, dtype=np.float32).reshape(shape)


def parse_outputs(text: str) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("output["):
            i += 1
            continue
        i += 1
        if i >= len(lines) or not lines[i].startswith("shape:"):
            raise ValueError(f"missing shape after output header: {lines[i-1:]}")
        shape_txt = lines[i].split(":", 1)[1].strip()
        i += 1
        if shape_txt == "scalar":
            shape: tuple[int, ...] = ()
        else:
            shape = tuple(int(d) for d in shape_txt.split("x") if d)
        values: list[float] = []
        needed = int(np.prod(shape)) if shape else 1
        while i < len(lines) and not lines[i].startswith("output[") and len(values) < needed:
            if lines[i].strip():
                values.extend(float(tok) for tok in lines[i].split())
            i += 1
        if len(values) != needed:
            raise ValueError(f"expected {needed} values, got {len(values)}")
        outputs.append(np.array(values, dtype=np.float32).reshape(shape))
    if not outputs:
        raise ValueError(f"no JIT outputs parsed from:\n{text}")
    return outputs


def reference(name: str) -> list[np.ndarray]:
    if name == "add":
        a = sequential((2, 2))
        return [a + a]
    if name == "relu":
        return [np.maximum(sequential((2, 2)), 0.0)]
    if name == "matmul":
        a = sequential((2, 4))
        b = sequential((4, 3))
        return [a @ b]
    if name == "linear":
        x = sequential((2, 4))
        w = np.arange(12, dtype=np.float32).reshape(4, 3)
        b = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        return [np.maximum(x @ w + b, 0.0)]
    raise ValueError(f"unknown model {name}")


def run_one(tc_compile: str, model: Path) -> None:
    proc = subprocess.run(
        [tc_compile, str(model), "-emit=jit"],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{model.name} jit failed:\n{proc.stderr or proc.stdout}"
        )
    got = parse_outputs(proc.stdout)
    exp = reference(model.stem)
    if len(got) != len(exp):
        raise AssertionError(f"{model.name}: {len(got)} outputs, expected {len(exp)}")
    for i, (g, e) in enumerate(zip(got, exp)):
        if not np.allclose(g, e, rtol=REL_TOL, atol=ABS_TOL):
            raise AssertionError(
                f"{model.name} output[{i}] mismatch\n got:\n{g}\n exp:\n{e}"
            )
    print(f"ok {model.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc-compile", default="tc-compile")
    parser.add_argument("--src", type=Path, required=True)
    args = parser.parse_args()
    models = args.src / "models"
    for name in ("add", "relu", "matmul", "linear"):
        run_one(args.tc_compile, models / f"{name}.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
