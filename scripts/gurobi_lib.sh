#!/usr/bin/env bash
# Manage the Gurobi native library used by the project venv.
#
# The gurobipy wheel bundles a CPU-only libgurobi130.so and silently ignores
# GRB_LIB, so switching builds requires swapping the .so inside the package
# directory. Both the CPU and CUDA (GPU) builds are available as tarballs in
# the repo root.
#
# Usage:
#   scripts/gurobi_lib.sh status   Show which build (CPU/GPU) is active
#   scripts/gurobi_lib.sh cpu      Install the CPU build (gurobi13.0.3_linux64.tar.gz)
#   scripts/gurobi_lib.sh cuda     Install the CUDA build (gurobi13.0.3_linux64cuda13.tar.gz)
#   scripts/gurobi_lib.sh lic      Install the repo's gurobi.lic as ~/.gurobi.lic
#
# Notes:
# - Without a full license, gurobipy falls back to the size-limited pip
#   license (2000 vars, 200 for quadratic models). The repo's WLS license
#   must be referenced explicitly: either via GRB_LICENSE_FILE (which
#   tests/conftest.py does automatically when gurobi.lic is in the repo root)
#   or by copying it to ~/.gurobi.lic and exporting GRB_LICENSE_FILE=$HOME/.gurobi.lic.
# - The GPU build needs an NVIDIA driver with a CUDA 13 compatible runtime.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
CPU_TARBALL="$REPO_ROOT/gurobi13.0.3_linux64.tar.gz"
CUDA_TARBALL="$REPO_ROOT/gurobi13.0.3_linux64cuda13.tar.gz"
PY="$VENV/bin/python"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -x "$PY" ] || die "venv not found at $VENV (set VENV=... to override)"

LIBDIR="$("$PY" -c 'import gurobipy, os; print(os.path.join(os.path.dirname(gurobipy.__file__), ".libs"))')"
SO="$LIBDIR/libgurobi130.so"

banner() {
  "$PY" - <<'EOF' 2>&1 | grep -iE "license|gurobi optimizer|cpu model|gpu model|gpu not|running on cpu"
import gurobipy as gp
env = gp.Env(empty=True)
env.start()
m = gp.Model(env=env)
m.addVar(lb=0, ub=1)
m.setParam("OutputFlag", 1)
m.optimize()
EOF
}

banner | grep -q "linux64gpu" && ACTIVE=cuda || ACTIVE=cpu

case "${1:-status}" in
  status)
    echo "venv:  $VENV"
    echo "lib:   $SO ($(du -h "$SO" | cut -f1))"
    echo "build: $ACTIVE"
    banner | sed 's/^/        /'
    ;;
  cpu|cuda)
    if [ "$ACTIVE" = "$1" ]; then
      echo "already running the $1 build"
      exit 0
    fi
    [ "${1}" = cuda ] && TARBALL="$CUDA_TARBALL" || TARBALL="$CPU_TARBALL"
    [ -f "$TARBALL" ] || die "tarball not found: $TARBALL"
    cp "$SO" "$SO.${ACTIVE}.bak"
    TMP="$(mktemp -d)"
    tar xzf "$TARBALL" -C "$TMP"
    cp "$TMP/gurobi1303/linux64$( [ "${1}" = cuda ] && echo cuda13 )/lib/libgurobi130.so" "$SO"
    rm -rf "$TMP"
    chmod +x "$SO"
    echo "installed $1 build (previous build kept at $SO.${ACTIVE}.bak)"
    banner | sed 's/^/        /'
    ;;
  lic)
    cp "$REPO_ROOT/gurobi.lic" "$HOME/.gurobi.lic"
    echo "installed $REPO_ROOT/gurobi.lic -> ~/.gurobi.lic"
    echo "note: also export GRB_LICENSE_FILE=\$HOME/.gurobi.lic (WLS files are not"
    echo "resolved from the default search path on their own)"
    ;;
  *)
    die "unknown command '$1' (expected status|cpu|cuda|lic)"
    ;;
esac
