#!/usr/bin/env bash
# Symlink the hpc-alloc CLI into ~/.local/bin and the Claude Code skill into ~/.claude/skills.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"

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
echo "Next: hpc-alloc setup --netid YOUR_NETID"
echo "      (creates ~/.config/hpc-alloc/config.toml — see config.example.toml for options)"
