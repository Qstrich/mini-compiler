#!/usr/bin/env bash
# CPU-only CI entry: build and run the lit suite (FileCheck + JIT numerics).
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/build.sh" check-tc
