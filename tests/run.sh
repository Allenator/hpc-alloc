#!/usr/bin/env bash
# Strict offline v2 suite. No cluster, network, credentials, or legacy state required.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"

cd "$repo"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git ls-files --error-unmatch hpc_alloc/__init__.py hpc_alloc/cli.py >/dev/null; then
    echo "tracked-delivery check failed: the hpc_alloc package is not in Git" >&2
    exit 1
  fi
  untracked_delivery_files="$(git ls-files --others --exclude-standard -- hpc_alloc tests)"
  if [[ -n "$untracked_delivery_files" ]]; then
    echo "tracked-delivery check failed: source or test files are untracked:" >&2
    echo "$untracked_delivery_files" >&2
    exit 1
  fi
fi
cache="$(mktemp -d)"
trap 'rm -rf "$cache"' EXIT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$cache"
export PYTHONWARNINGS=error

python3 -m unittest discover -s tests -t . -v
python3 -m py_compile hpc-alloc hpc_alloc/*.py

echo "all hpc-alloc v2 tests passed"
