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

# Keep this explicit: installation must never succeed from a partial source tree
# merely because the command path used during preflight did not import a missing
# runtime module.
runtime_modules=(
  hpc_alloc
  hpc_alloc.cli
  hpc_alloc.commands
  hpc_alloc.config
  hpc_alloc.context
  hpc_alloc.eligibility
  hpc_alloc.errors
  hpc_alloc.lifecycle
  hpc_alloc.locking
  hpc_alloc.models
  hpc_alloc.monitor
  hpc_alloc.ownership
  hpc_alloc.output
  hpc_alloc.paths
  hpc_alloc.retry
  hpc_alloc.schedulability
  hpc_alloc.selectors
  hpc_alloc.slurm
  hpc_alloc.ssh
  hpc_alloc.ssh_config
  hpc_alloc.state
  hpc_alloc.streaming
)

skill_files=(
  skill/SKILL.md
  skill/references/command-contracts.md
  skill/references/recovery-and-lifecycle.md
)

if [[ ! -f "$here/hpc_alloc/__init__.py" ]]; then
  echo "hpc-alloc installation is incomplete: $here/hpc_alloc is missing" >&2
  exit 1
fi
if [[ ! -f "$here/hpc-alloc" ]]; then
  echo "hpc-alloc installation is incomplete: $here/hpc-alloc is missing" >&2
  exit 1
fi
for source in "${skill_files[@]}"; do
  if [[ ! -f "$here/$source" ]]; then
    echo "hpc-alloc installation is incomplete: $here/$source is missing" >&2
    exit 1
  fi
done
if ! python3 -I -B -c '
import importlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
package = root / "hpc_alloc"
manifest = tuple(sys.argv[2:])

sources = {}
for name in manifest:
    if name == "hpc_alloc":
        source = "__init__.py"
    elif name.startswith("hpc_alloc."):
        leaf = name.removeprefix("hpc_alloc.")
        if not leaf.isidentifier() or "." in leaf:
            raise SystemExit(1)
        source = f"{leaf}.py"
    else:
        raise SystemExit(1)
    if name in sources or source in sources.values():
        raise SystemExit(1)
    sources[name] = source

actual_sources = {path.name for path in package.glob("*.py") if path.is_file()}
if set(sources.values()) != actual_sources:
    raise SystemExit(1)

sys.path.insert(0, str(root))
for name, source in sources.items():
    try:
        module = importlib.import_module(name)
        module_path = pathlib.Path(module.__file__).resolve()
    except BaseException:
        raise SystemExit(1)
    if not module_path.is_relative_to(package):
        raise SystemExit(1)
    if module_path != (package / source).resolve():
        raise SystemExit(1)
' "$here" "${runtime_modules[@]}"; then
  echo "hpc-alloc installation is incomplete: the adjacent Python package cannot be imported or does not match the runtime-module manifest" >&2
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
# `ln -sfn SRC DIR` descends into an existing real directory and creates
# DIR/skill inside it, so SKILL.md would land at hpc-alloc/skill/SKILL.md and
# never load -- while the install still reported success.  Clear a non-symlink
# target first so the link always replaces it.
if [ -e "$skills_dir/hpc-alloc" ] && [ ! -L "$skills_dir/hpc-alloc" ]; then
  rm -rf "$skills_dir/hpc-alloc"
fi
ln -sfn "$here/skill" "$skills_dir/hpc-alloc"
echo "linked $skills_dir/hpc-alloc -> $here/skill"

echo
echo "hpc-alloc v2 requires a new authoritative config and SQLite state database."
echo "Next: hpc-alloc setup --netid YOUR_NETID"
echo "      (use --force only to replace an existing config; v1 state is not imported)"
