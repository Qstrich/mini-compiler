"""Pytest goldens: JIT vs NumPy/ORT."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
import check_numeric as cn  # noqa: E402


def _tc_compile() -> str:
    env = os.environ.get("TC_COMPILE")
    if env:
        return env
    built = ROOT / "build" / "bin" / "tc-compile"
    if built.is_file():
        return str(built)
    found = shutil.which("tc-compile")
    if found:
        return found
    pytest.skip("tc-compile not found (build the project or set TC_COMPILE)")


@pytest.mark.parametrize("name", cn.MODELS)
@pytest.mark.parametrize("extra", [None, ["-tile-sizes=2,2,2"]])
def test_jit_matches_numpy(name: str, extra: list[str] | None) -> None:
    cn.run_one(_tc_compile(), ROOT / "models" / f"{name}.onnx", extra)
