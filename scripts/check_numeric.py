#!/usr/bin/env python3
"""Compare tc-compile -emit=jit output to NumPy, and ONNX Runtime if present."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REL_TOL = 1e-4
ABS_TOL = 1e-5

MODELS = ("add", "relu", "matmul", "linear")


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


def feeds(name: str) -> dict[str, np.ndarray]:
    if name == "add":
        a = sequential((2, 2))
        return {"A": a, "B": a}
    if name == "relu":
        return {"X": sequential((2, 2))}
    if name == "matmul":
        return {"A": sequential((2, 4)), "B": sequential((4, 3))}
    if name == "linear":
        return {"X": sequential((2, 4))}
    raise ValueError(f"unknown model {name}")


def numpy_reference(name: str) -> list[np.ndarray]:
    ins = feeds(name)
    if name == "add":
        return [ins["A"] + ins["B"]]
    if name == "relu":
        return [np.maximum(ins["X"], 0.0)]
    if name == "matmul":
        return [ins["A"] @ ins["B"]]
    if name == "linear":
        w = np.arange(12, dtype=np.float32).reshape(4, 3)
        b = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        return [np.maximum(ins["X"] @ w + b, 0.0)]
    raise ValueError(f"unknown model {name}")


def try_ort_reference(model: Path, name: str) -> list[np.ndarray] | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds(name))
    return [np.asarray(o, dtype=np.float32) for o in outs]


def assert_close(label: str, got: list[np.ndarray], exp: list[np.ndarray]) -> None:
    if len(got) != len(exp):
        raise AssertionError(f"{label}: {len(got)} outputs, expected {len(exp)}")
    for i, (g, e) in enumerate(zip(got, exp)):
        if not np.allclose(g, e, rtol=REL_TOL, atol=ABS_TOL):
            raise AssertionError(f"{label} output[{i}] mismatch\n got:\n{g}\n exp:\n{e}")


def run_jit(tc_compile: str, model: Path, extra: list[str] | None = None) -> str:
    cmd = [tc_compile, str(model), "-emit=jit"]
    if extra:
        cmd.extend(extra)
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{model.name} jit failed:\n{proc.stderr or proc.stdout}")
    return proc.stdout


def run_one(
    tc_compile: str,
    model: Path,
    extra: list[str] | None = None,
    require_ort: bool = False,
) -> None:
    got = parse_outputs(run_jit(tc_compile, model, extra))
    label = model.name if not extra else f"{model.name} {' '.join(extra)}"
    assert_close(f"{label} numpy", got, numpy_reference(model.stem))
    ort = try_ort_reference(model, model.stem)
    if ort is None:
        if require_ort:
            raise RuntimeError("onnxruntime is required but not installed")
        print(f"ok {label} (numpy)")
        return
    assert_close(f"{label} ort", got, ort)
    print(f"ok {label} (numpy+ort)")


def run_all(tc_compile: str, src: Path, require_ort: bool = False) -> None:
    models = src / "models"
    for name in MODELS:
        path = models / f"{name}.onnx"
        run_one(tc_compile, path, require_ort=require_ort)
        run_one(tc_compile, path, ["-tile-sizes=2,2,2"], require_ort=require_ort)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc-compile", default="tc-compile")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument(
        "--require-ort",
        action="store_true",
        help="Fail if onnxruntime is not installed",
    )
    args = parser.parse_args()
    run_all(args.tc_compile, args.src, require_ort=args.require_ort)
    return 0


if __name__ == "__main__":
    sys.exit(main())
