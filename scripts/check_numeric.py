#!/usr/bin/env python3
"""Run each model in models.py through `tc-compile -emit=jit` and compare the
result to its NumPy reference, and to ONNX Runtime when it is installed."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from models import MODELS, Model, Shape

REL_TOL = 1e-4
ABS_TOL = 1e-5


def sequential(shape: Shape) -> np.ndarray:
    """What tc-compile feeds @main when no -input= is given: 0, 1, 2, ..."""
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)


def parse_outputs(text: str) -> list[np.ndarray]:
    """Parse `output[i]` / `shape:AxB` / rows-of-floats blocks."""
    outputs = []
    for block in text.split("output[")[1:]:
        header, *rows = block.splitlines()[1:]
        dims = header.removeprefix("shape:").strip()
        shape = () if dims == "scalar" else tuple(int(d) for d in dims.split("x"))
        values = [float(tok) for row in rows for tok in row.split()]
        outputs.append(np.array(values, dtype=np.float32).reshape(shape))
    if not outputs:
        raise ValueError(f"no JIT outputs parsed from:\n{text}")
    return outputs


def ort_outputs(path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray] | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return [np.asarray(o, dtype=np.float32) for o in session.run(None, feeds)]


def assert_close(label: str, got: list[np.ndarray], expected: list[np.ndarray]) -> None:
    if len(got) != len(expected):
        raise AssertionError(f"{label}: {len(got)} outputs, expected {len(expected)}")
    for i, (g, e) in enumerate(zip(got, expected)):
        if not np.allclose(g, e, rtol=REL_TOL, atol=ABS_TOL):
            raise AssertionError(f"{label} output[{i}] mismatch\n got:\n{g}\n expected:\n{e}")


def check(tc_compile: str, models_dir: Path, model: Model, require_ort: bool) -> None:
    path = models_dir / f"{model.name}.onnx"
    proc = subprocess.run([tc_compile, str(path), "-emit=jit"], text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{path.name} jit failed:\n{proc.stderr or proc.stdout}")
    got = parse_outputs(proc.stdout)

    feeds = {name: sequential(shape) for name, shape in model.inputs.items()}
    assert_close(f"{model.name} numpy", got, [model.reference({**feeds, **model.initializers})])

    ort = ort_outputs(path, feeds)
    if ort is None:
        if require_ort:
            raise RuntimeError("onnxruntime is required but not installed")
        print(f"ok {model.name} (numpy)")
        return
    assert_close(f"{model.name} ort", got, ort)
    print(f"ok {model.name} (numpy+ort)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tc-compile", default="tc-compile")
    parser.add_argument("--src", type=Path, required=True, help="repository root")
    parser.add_argument("--require-ort", action="store_true",
                        help="fail if onnxruntime is not installed")
    args = parser.parse_args()
    for model in MODELS:
        check(args.tc_compile, args.src / "models", model, args.require_ort)
    return 0


if __name__ == "__main__":
    sys.exit(main())
