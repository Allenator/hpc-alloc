from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXCLUDED_FALLBACK_DIRECTORIES = frozenset(
    {
        ".cache",
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)

_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
_BLOCKQUOTE_PREFIX = re.compile(r"^(?P<prefix>(?: {0,3}>[ \t]?)+)(?P<body>.*)$")
_FENCE_OPEN = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_HTML_TAG = re.compile(r"^ {0,3}</?[A-Za-z][A-Za-z0-9-]*(?:[ \t][^>]*)?>[ \t]*$")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]+|$)")
_REFERENCE_DEFINITION = re.compile(r"^ {0,3}\[[^]]+\]:[ \t]*\S")
_SETEXT_HEADING = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_TABLE_SEPARATOR = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$"
)
_THEMATIC_BREAK = re.compile(
    r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$"
)
_GIT_REPOSITORY_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    }
)


@dataclass(frozen=True)
class MarkdownViolation:
    line: int
    message: str


def _isolated_git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in _GIT_REPOSITORY_ENVIRONMENT:
        environment.pop(name, None)
    return environment


def discover_markdown_files(root: Path) -> tuple[Path, ...]:
    """Return tracked Markdown at a Git root, or Markdown in a clean export."""

    root = root.resolve()
    try:
        git_root_result = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "--show-toplevel"],
            env=_isolated_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        git_root_result = None

    if git_root_result is not None and git_root_result.returncode == 0:
        rendered_git_root = os.fsdecode(git_root_result.stdout).strip()
        if rendered_git_root and Path(rendered_git_root).resolve() == root:
            tracked_result = subprocess.run(
                ["git", "-C", os.fspath(root), "ls-files", "-z", "--"],
                env=_isolated_git_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if tracked_result.returncode != 0:
                detail = os.fsdecode(tracked_result.stderr).strip()
                raise AssertionError(f"could not list tracked Markdown files: {detail}")
            relative_paths = (
                Path(os.fsdecode(item))
                for item in tracked_result.stdout.split(b"\0")
                if item and os.fsdecode(item).lower().endswith(".md")
            )
            return tuple(
                root / relative_path
                for relative_path in sorted(
                    relative_paths, key=lambda path: os.fsencode(os.fspath(path))
                )
            )

    discovered = []
    for directory, child_directories, filenames in os.walk(root):
        child_directories[:] = sorted(
            name
            for name in child_directories
            if name not in EXCLUDED_FALLBACK_DIRECTORIES
        )
        for filename in filenames:
            if not filename.lower().endswith(".md"):
                continue
            path = Path(directory, filename)
            if path.is_file():
                discovered.append(path)
    return tuple(sorted(discovered, key=lambda path: path.relative_to(root).as_posix()))


def _blockquote_body(line: str) -> tuple[int, str] | None:
    match = _BLOCKQUOTE_PREFIX.match(line)
    if match is None:
        return None
    return match.group("prefix").count(">"), match.group("body")


def _fence(line: str) -> str | None:
    quoted = _blockquote_body(line)
    candidate = quoted[1] if quoted is not None else line
    match = _FENCE_OPEN.match(candidate)
    if match is None:
        return None
    fence = match.group("fence")
    if fence.startswith("`") and "`" in match.group("info"):
        return None
    return fence


def _is_fence_close(line: str, character: str, minimum_length: int) -> bool:
    quoted = _blockquote_body(line)
    candidate = quoted[1] if quoted is not None else line
    return bool(
        re.fullmatch(
            rf"[ \t]*{re.escape(character)}{{{minimum_length},}}[ \t]*",
            candidate,
        )
    )


def _table_lines(lines: list[str]) -> set[int]:
    table_lines: set[int] = set()
    for index, line in enumerate(lines):
        if not _TABLE_SEPARATOR.match(line):
            continue
        table_lines.add(index)
        if index > 0 and lines[index - 1].strip():
            table_lines.add(index - 1)
        following = index + 1
        while following < len(lines) and lines[following].strip():
            if "|" not in lines[following]:
                break
            table_lines.add(following)
            following += 1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.add(index)
    return table_lines


def markdown_style_violations(text: str) -> tuple[MarkdownViolation, ...]:
    """Find source-width wrapping while preserving Markdown structural lines."""

    lines = text.splitlines()
    table_lines = _table_lines(lines)
    violations: list[MarkdownViolation] = []
    in_frontmatter = bool(lines and lines[0].strip() == "---")
    frontmatter_closed = not in_frontmatter
    in_html_comment = False
    fence_character: str | None = None
    fence_length = 0
    in_indented_code = False
    previous_kind: str | None = None
    previous_blank = True

    for index, line in enumerate(lines):
        line_number = index + 1
        stripped = line.strip()

        if in_frontmatter:
            if index > 0 and stripped in {"---", "..."}:
                in_frontmatter = False
                frontmatter_closed = True
            continue

        if fence_character is not None:
            if _is_fence_close(line, fence_character, fence_length):
                fence_character = None
                fence_length = 0
            continue
        candidate_fence = _fence(line)
        if candidate_fence is not None:
            fence_character = candidate_fence[0]
            fence_length = len(candidate_fence)
            previous_kind = "structure"
            previous_blank = False
            in_indented_code = False
            continue

        if in_html_comment:
            if "-->" in line:
                in_html_comment = False
            continue
        if stripped.startswith("<!--"):
            in_html_comment = "-->" not in line
            previous_kind = "structure"
            previous_blank = False
            continue

        if not stripped:
            previous_kind = None
            previous_blank = True
            continue

        begins_indented_code = line.startswith("\t") or len(line) - len(
            line.lstrip(" ")
        ) >= 4
        if begins_indented_code and (
            in_indented_code
            or previous_blank
            or previous_kind == "structure"
            or index == 0
        ):
            in_indented_code = True
            previous_kind = "structure"
            previous_blank = False
            continue
        in_indented_code = False

        if line.endswith("  "):
            violations.append(
                MarkdownViolation(
                    line_number,
                    "replace the two-space hard break with inline <br> on the same physical line",
                )
            )
        trailing_backslashes = len(line) - len(line.rstrip("\\"))
        if trailing_backslashes % 2 == 1:
            violations.append(
                MarkdownViolation(
                    line_number,
                    "replace the backslash hard break with inline <br> on the same physical line",
                )
            )

        quoted = _blockquote_body(line)
        quote_depth = quoted[0] if quoted is not None else 0
        content = quoted[1] if quoted is not None else line
        content_stripped = content.strip()

        is_list_item = bool(_LIST_ITEM.match(content))
        is_structure = (
            index in table_lines
            or bool(_ATX_HEADING.match(content))
            or bool(_SETEXT_HEADING.match(content))
            or bool(_THEMATIC_BREAK.match(content))
            or bool(_REFERENCE_DEFINITION.match(content))
            or bool(_HTML_TAG.match(content))
            or (quoted is not None and not content_stripped)
        )

        if is_structure:
            current_kind = "structure"
        elif is_list_item:
            current_kind = f"list:{quote_depth}"
        elif quoted is not None:
            current_kind = f"quote:{quote_depth}"
        else:
            current_kind = "prose"

        if current_kind == "prose" and previous_kind in {"prose", "list:0"}:
            subject = "list item" if previous_kind == "list:0" else "prose paragraph"
            violations.append(
                MarkdownViolation(
                    line_number,
                    f"join this physical line to the preceding {subject}",
                )
            )
        elif current_kind.startswith("quote:") and previous_kind in {
            current_kind,
            current_kind.replace("quote:", "list:", 1),
        }:
            subject = (
                "list item" if previous_kind.startswith("list:") else "blockquote paragraph"
            )
            violations.append(
                MarkdownViolation(
                    line_number,
                    f"join this physical line to the preceding {subject}",
                )
            )

        previous_kind = current_kind
        previous_blank = False

    if in_frontmatter and not frontmatter_closed:
        violations.append(MarkdownViolation(1, "close the YAML frontmatter"))
    if fence_character is not None:
        violations.append(MarkdownViolation(len(lines) or 1, "close the fenced code block"))
    if in_html_comment:
        violations.append(MarkdownViolation(len(lines) or 1, "close the HTML comment"))
    return tuple(violations)


class MarkdownDiscoveryTests(unittest.TestCase):
    def test_exact_git_root_uses_nul_delimited_tracked_files_only(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "--quiet", os.fspath(root)],
                env=_isolated_git_environment(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tracked = root / "tracked.md"
            uppercase = root / "GUIDE.MD"
            unusual = root / "line\nbreak.md"
            untracked = root / "untracked.md"
            tracked.write_text("Tracked paragraph.\n", encoding="utf-8")
            uppercase.write_text("Uppercase extension.\n", encoding="utf-8")
            unusual.write_text("Unusual tracked paragraph.\n", encoding="utf-8")
            untracked.write_text("This is\nwrapped.\n", encoding="utf-8")
            subprocess.run(
                [
                    "git",
                    "-C",
                    os.fspath(root),
                    "add",
                    "--",
                    tracked.name,
                    uppercase.name,
                    unusual.name,
                ],
                env=_isolated_git_environment(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            discovered = discover_markdown_files(root)

            self.assertEqual(
                set(discovered),
                {tracked.resolve(), uppercase.resolve(), unusual.resolve()},
            )
            self.assertNotIn(untracked.resolve(), discovered)

    def test_clean_export_fallback_recurses_but_ignores_metadata_and_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = {
                root / "README.md",
                root / "skill" / "SKILL.md",
                root / "skill" / "REFERENCE.MD",
            }
            excluded = {
                root / ".git" / "hidden.md",
                root / ".cache" / "cached.md",
                root / "node_modules" / "package.md",
                root / "__pycache__" / "generated.md",
            }
            for path in included | excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("One paragraph.\n", encoding="utf-8")

            self.assertEqual(
                set(discover_markdown_files(root)),
                {path.resolve() for path in included},
            )

    def test_nested_directory_is_not_treated_as_the_parent_git_index(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            subprocess.run(
                ["git", "init", "--quiet", os.fspath(parent)],
                env=_isolated_git_environment(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            export = parent / "clean-export"
            export.mkdir()
            document = export / "README.md"
            document.write_text("Export paragraph.\n", encoding="utf-8")

            self.assertEqual(discover_markdown_files(export), (document.resolve(),))


class MarkdownStyleTests(unittest.TestCase):
    def test_structural_markdown_and_one_line_prose_are_accepted(self) -> None:
        compliant = """---
name: example
description: This frontmatter value may be handled by YAML rules.
---
# Heading

Each prose paragraph occupies one source line even when it is deliberately long.

- Each bullet occupies one source line.
  - Nested bullets are separate structural lines.
1. Ordered items work too.
12. Multi-digit ordered items are recognized.

> Each blockquote paragraph occupies one source line.
>
> A second paragraph follows a quoted blank line.

| Column A | Column B |
|---|---|
| Value | Value |

Column A | Column B
--- | ---
Value | Value

Setext heading
--------------

[reference]: https://example.invalid/document

<!--
Comments may use structural source lines.
-->

<details>
<summary>Structural HTML remains separate.</summary>

The prose inside an HTML block still occupies one line.

</details>

```bash
printf '%s\\n' 'fenced code may wrap' \\
  'without becoming prose'
```

  ~~~text
  ~~~ text inside the fence is not a closing delimiter
  An indented fence is structural.
  ~~~

    printf '%s\\n' 'indented code may end in a shell continuation' \\
        'and may span physical lines'
"""
        self.assertEqual(markdown_style_violations(compliant), ())

    def test_wrapped_prose_lists_and_blockquotes_are_rejected(self) -> None:
        cases = {
            "paragraph": ("First half of a paragraph\nsecond half.\n", 2, "prose paragraph"),
            "bullet": ("- First half of an item\n  second half.\n", 2, "list item"),
            "ordered": ("12. First half of an item\n    second half.\n", 2, "list item"),
            "blockquote": ("> First half of a quote\n> second half.\n", 2, "blockquote"),
            "quoted list": ("> - First half of an item\n>   second half.\n", 2, "list item"),
        }
        for label, (source, line, message) in cases.items():
            with self.subTest(label=label):
                violations = markdown_style_violations(source)
                self.assertTrue(
                    any(item.line == line and message in item.message for item in violations),
                    violations,
                )

    def test_unquoted_list_followed_by_blockquote_is_not_a_wrapped_item(self) -> None:
        source = "- A complete unquoted item.\n> A separate blockquote paragraph.\n"
        self.assertEqual(markdown_style_violations(source), ())

    def test_source_level_hard_breaks_are_rejected_outside_code(self) -> None:
        cases = {
            "spaces": "A line with a hard break.  \n",
            "backslash": "A line with a hard break.\\\n",
        }
        for label, source in cases.items():
            with self.subTest(label=label):
                violations = markdown_style_violations(source)
                self.assertTrue(
                    any(
                        "hard break" in item.message
                        and "inline <br> on the same physical line" in item.message
                        for item in violations
                    )
                )

    def test_all_repository_markdown_uses_one_source_line_per_prose_unit(self) -> None:
        failures: list[str] = []
        for path in discover_markdown_files(REPO):
            relative_path = path.relative_to(REPO)
            if not path.is_file():
                failures.append(f"{relative_path}: tracked Markdown file is missing")
                continue
            source = path.read_text(encoding="utf-8")
            for violation in markdown_style_violations(source):
                failures.append(f"{relative_path}:{violation.line}: {violation.message}")
        self.assertEqual(failures, [], "\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
