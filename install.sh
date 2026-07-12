#!/usr/bin/env bash
# Install hpc-alloc v2 by linking the CLI and bundled Claude Code skill.
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "hpc-alloc requires Python 3.11 or newer; python3 was not found" >&2
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  version="$(python3 -c 'import platform; print(platform.python_version())')"
  echo "hpc-alloc requires Python 3.11 or newer; found Python $version" >&2
  exit 1
fi

here="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "$here/hpc_alloc/__init__.py" ]]; then
  echo "hpc-alloc installation is incomplete: $here/hpc_alloc is missing" >&2
  exit 1
fi
if [[ ! -f "$here/hpc-alloc" ]]; then
  echo "hpc-alloc installation is incomplete: $here/hpc-alloc is missing" >&2
  exit 1
fi
if [[ ! -f "$here/skill/SKILL.md" ]]; then
  echo "hpc-alloc installation is incomplete: $here/skill/SKILL.md is missing" >&2
  exit 1
fi
if ! python3 -I -B -c '
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
try:
    import hpc_alloc.cli as cli
except Exception:
    raise SystemExit(1)
module = pathlib.Path(cli.__file__).resolve()
raise SystemExit(0 if module.is_relative_to(root / "hpc_alloc") else 1)
' "$here"; then
  echo "hpc-alloc installation is incomplete: the adjacent Python package cannot be imported" >&2
  exit 1
fi

bin_dir="${HOME}/.local/bin"
mkdir -p "$bin_dir"
ln -sf "$here/hpc-alloc" "$bin_dir/hpc-alloc"
echo "linked $bin_dir/hpc-alloc -> $here/hpc-alloc"
case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *) echo "note: $bin_dir is not on your PATH — add it to your shell profile" ;;
esac

skills_dir="${HOME}/.claude/skills"
mkdir -p "$skills_dir"
ln -sfn "$here/skill" "$skills_dir/hpc-alloc"
echo "linked $skills_dir/hpc-alloc -> $here/skill"

echo
echo "hpc-alloc v2 requires a new authoritative config and SQLite state database."
echo "Next: hpc-alloc setup --netid YOUR_NETID"
echo "      (use --force only to replace an existing config; v1 state is not imported)"
