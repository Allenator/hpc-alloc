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
_BLOCKQUOTE_MARKER = re.compile(r"^ {0,3}> ?")
_FENCE_OPEN = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_HTML_TAG = re.compile(r"^ {0,3}</?[A-Za-z][A-Za-z0-9-]*(?:[ \t][^>]*)?>[ \t]*$")
_LIST_MARKER = re.compile(
    r"^(?P<indent> {0,3})(?P<marker>[-+*]|(?P<number>[0-9]{1,9})[.)])"
)
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


@dataclass(frozen=True)
class _ContainerFrame:
    kind: str
    identity: tuple[int, int]
    content_indent: int = 0
    marker_indent: int = 0


@dataclass(frozen=True)
class _ListMarkerMatch:
    end: int
    content_indent: int
    marker_indent: int
    has_content: bool
    interrupts_paragraph: bool


@dataclass(frozen=True)
class _OpenedContainer:
    kind: str
    identity: tuple[int, int]
    interrupts_paragraph: bool
    empty_item: bool = False


@dataclass(frozen=True)
class _ParsedLine:
    frames: tuple[_ContainerFrame, ...]
    content: str
    opened: tuple[_OpenedContainer, ...]


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


def _fence(content: str) -> str | None:
    match = _FENCE_OPEN.match(content)
    if match is None:
        return None
    fence = match.group("fence")
    if fence.startswith("`") and "`" in match.group("info"):
        return None
    return fence


def _is_fence_close(content: str, character: str, minimum_length: int) -> bool:
    return bool(
        re.fullmatch(
            rf" {{0,3}}{re.escape(character)}{{{minimum_length},}}[ \t]*",
            content,
        )
    )


def _match_list_marker(content: str) -> _ListMarkerMatch | None:
    match = _LIST_MARKER.match(content)
    if match is None:
        return None
    marker_end = match.end()
    if marker_end < len(content) and content[marker_end] != " ":
        return None

    after_marker = content[marker_end:]
    following_spaces = len(after_marker) - len(after_marker.lstrip(" "))
    if not after_marker.strip():
        padding = 1
        content_start = len(content)
    elif following_spaces <= 4:
        padding = following_spaces
        content_start = marker_end + following_spaces
    else:
        padding = 1
        content_start = marker_end + 1

    marker = match.group("marker")
    number = match.group("number")
    has_content = bool(content[content_start:].strip())
    interrupts = has_content and (number is None or int(number) == 1)
    return _ListMarkerMatch(
        end=content_start,
        content_indent=len(match.group("indent")) + len(marker) + padding,
        marker_indent=len(match.group("indent")),
        has_content=has_content,
        interrupts_paragraph=interrupts,
    )


def _parse_containers(
    line: str,
    active_frames: tuple[_ContainerFrame, ...],
    line_number: int,
    paragraph_frames: tuple[_ContainerFrame, ...] | None,
) -> _ParsedLine:
    """Peel explicit and indented container prefixes from one expanded line."""

    cursor = 0
    frames: list[_ContainerFrame] = []
    opened: list[_OpenedContainer] = []
    existing_index = 0

    while existing_index < len(active_frames):
        frame = active_frames[existing_index]
        remainder = line[cursor:]
        if frame.kind == "quote":
            marker = _BLOCKQUOTE_MARKER.match(remainder)
            if marker is None:
                break
            cursor += marker.end()
            frames.append(frame)
            existing_index += 1
            continue

        marker = None
        if not _THEMATIC_BREAK.match(remainder):
            marker = _match_list_marker(remainder)
        if marker is not None and marker.marker_indent == frame.marker_indent:
            replacement = _ContainerFrame(
                kind="list",
                identity=(line_number, len(frames)),
                content_indent=marker.content_indent,
                marker_indent=marker.marker_indent,
            )
            cursor += marker.end
            frames.append(replacement)
            opened.append(
                _OpenedContainer(
                    kind="list",
                    identity=replacement.identity,
                    interrupts_paragraph=True,
                    empty_item=not marker.has_content,
                )
            )
            break

        leading_spaces = len(remainder) - len(remainder.lstrip(" "))
        if leading_spaces < frame.content_indent:
            break
        cursor += frame.content_indent
        frames.append(frame)
        existing_index += 1

    while True:
        remainder = line[cursor:]
        quote = _BLOCKQUOTE_MARKER.match(remainder)
        if quote is not None:
            frame = _ContainerFrame(
                kind="quote",
                identity=(line_number, len(frames)),
            )
            frames.append(frame)
            opened.append(
                _OpenedContainer(
                    kind="quote",
                    identity=frame.identity,
                    interrupts_paragraph=True,
                )
            )
            cursor += quote.end()
            continue

        setext_for_open_paragraph = (
            paragraph_frames is not None
            and tuple(frames) == paragraph_frames
            and bool(_SETEXT_HEADING.match(remainder))
        )
        if _THEMATIC_BREAK.match(remainder) or setext_for_open_paragraph:
            break
        marker = _match_list_marker(remainder)
        if marker is None:
            break
        frame = _ContainerFrame(
            kind="list",
            identity=(line_number, len(frames)),
            content_indent=marker.content_indent,
            marker_indent=marker.marker_indent,
        )
        frames.append(frame)
        opened.append(
            _OpenedContainer(
                kind="list",
                identity=frame.identity,
                interrupts_paragraph=marker.interrupts_paragraph,
                empty_item=not marker.has_content,
            )
        )
        cursor += marker.end

    return _ParsedLine(tuple(frames), line[cursor:], tuple(opened))


def _strip_container_frames(
    line: str, frames: tuple[_ContainerFrame, ...]
) -> tuple[tuple[_ContainerFrame, ...], str]:
    """Return the matched container prefix and its residual content."""

    cursor = 0
    matched_frames: list[_ContainerFrame] = []
    for frame in frames:
        remainder = line[cursor:]
        if frame.kind == "quote":
            marker = _BLOCKQUOTE_MARKER.match(remainder)
            if marker is None:
                break
            cursor += marker.end()
        else:
            if not remainder.strip():
                matched_frames.append(frame)
                continue
            leading_spaces = len(remainder) - len(remainder.lstrip(" "))
            if leading_spaces < frame.content_indent:
                break
            cursor += frame.content_indent
        matched_frames.append(frame)
    return tuple(matched_frames), line[cursor:]


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
    html_comment_frames: tuple[_ContainerFrame, ...] = ()
    fence_character: str | None = None
    fence_length = 0
    fence_frames: tuple[_ContainerFrame, ...] = ()
    indented_code_frames: tuple[_ContainerFrame, ...] | None = None
    active_frames: tuple[_ContainerFrame, ...] = ()
    paragraph_frames: tuple[_ContainerFrame, ...] | None = None
    paragraph_subject = "prose paragraph"
    pending_empty_items: set[tuple[int, int]] = set()

    for index, line in enumerate(lines):
        line_number = index + 1
        stripped = line.strip()
        expanded = line.expandtabs(4)

        if in_frontmatter:
            if index > 0 and stripped in {"---", "..."}:
                in_frontmatter = False
                frontmatter_closed = True
            continue

        if fence_character is not None:
            matched_frames, candidate = _strip_container_frames(
                expanded, fence_frames
            )
            if matched_frames == fence_frames:
                if _is_fence_close(candidate, fence_character, fence_length):
                    fence_character = None
                    fence_length = 0
                    active_frames = fence_frames
                    fence_frames = ()
                continue
            fence_character = None
            fence_length = 0
            fence_frames = ()
            active_frames = matched_frames

        if in_html_comment:
            matched_frames, candidate = _strip_container_frames(
                expanded, html_comment_frames
            )
            if matched_frames == html_comment_frames:
                if "-->" in candidate:
                    in_html_comment = False
                    html_comment_frames = ()
                continue
            in_html_comment = False
            html_comment_frames = ()
            active_frames = matched_frames

        if not stripped:
            pending_indexes = [
                frame_index
                for frame_index, frame in enumerate(active_frames)
                if frame.identity in pending_empty_items
            ]
            if pending_indexes:
                active_frames = active_frames[: min(pending_indexes)]
                retained_identities = {frame.identity for frame in active_frames}
                pending_empty_items.intersection_update(retained_identities)
            paragraph_frames = None
            indented_code_frames = None
            continue

        parsed = _parse_containers(
            expanded,
            active_frames,
            line_number,
            paragraph_frames,
        )
        content = parsed.content
        content_stripped = content.strip()
        prior_pending_items = pending_empty_items.copy()
        parsed_identities = {frame.identity for frame in parsed.frames}
        pending_empty_items.intersection_update(parsed_identities)
        if content_stripped or parsed.opened:
            pending_empty_items.difference_update(prior_pending_items)
        pending_empty_items.update(
            opened.identity
            for opened in parsed.opened
            if opened.kind == "list" and opened.empty_item
        )
        candidate_fence = _fence(content)
        opens_html_comment = content.lstrip().startswith("<!--")
        is_reference_definition = bool(_REFERENCE_DEFINITION.match(content))
        is_standalone_structure = (
            index in table_lines
            or bool(_ATX_HEADING.match(content))
            or bool(_SETEXT_HEADING.match(content))
            or bool(_THEMATIC_BREAK.match(content))
            or is_reference_definition
            or bool(_HTML_TAG.match(content))
            or bool(_TABLE_SEPARATOR.match(content))
            or (
                content_stripped.startswith("|")
                and content_stripped.endswith("|")
            )
            or not content_stripped
            or opens_html_comment
        )

        if paragraph_frames is not None:
            retained_frame_count = 0
            for parsed_frame, paragraph_frame in zip(
                parsed.frames, paragraph_frames
            ):
                if parsed_frame != paragraph_frame:
                    break
                retained_frame_count += 1
            dropped_paragraph_container = retained_frame_count < len(
                paragraph_frames
            )
            containers_interrupt = False
            if parsed.frames != paragraph_frames and parsed.opened:
                containers_interrupt = (
                    dropped_paragraph_container
                    or parsed.opened[0].interrupts_paragraph
                )
            leaf_interrupts_paragraph = candidate_fence is not None or (
                is_standalone_structure and not is_reference_definition
            )
            if parsed.frames == paragraph_frames:
                interrupts_paragraph = leaf_interrupts_paragraph
            elif parsed.opened:
                interrupts_paragraph = containers_interrupt
            else:
                interrupts_paragraph = leaf_interrupts_paragraph

            if not interrupts_paragraph:
                violations.append(
                    MarkdownViolation(
                        line_number,
                        f"join this physical line to the preceding {paragraph_subject}",
                    )
                )
                active_frames = paragraph_frames
                pending_empty_items.intersection_update(
                    frame.identity for frame in paragraph_frames
                )
                indented_code_frames = None
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
                continue
            paragraph_frames = None

        leading_content_spaces = len(content) - len(content.lstrip(" "))
        if leading_content_spaces >= 4 and (
            indented_code_frames is None or indented_code_frames == parsed.frames
        ):
            indented_code_frames = parsed.frames
            active_frames = parsed.frames
            continue
        indented_code_frames = None

        if candidate_fence is not None:
            fence_character = candidate_fence[0]
            fence_length = len(candidate_fence)
            fence_frames = parsed.frames
            active_frames = parsed.frames
            continue

        if opens_html_comment:
            in_html_comment = "-->" not in content
            html_comment_frames = parsed.frames if in_html_comment else ()
            active_frames = parsed.frames
            continue

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

        active_frames = parsed.frames
        if is_standalone_structure:
            continue

        paragraph_frames = parsed.frames
        if parsed.frames and parsed.frames[-1].kind == "list":
            paragraph_subject = "list item"
        elif parsed.frames and parsed.frames[-1].kind == "quote":
            paragraph_subject = "blockquote paragraph"
        else:
            paragraph_subject = "prose paragraph"

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

    def test_container_indented_code_fences_and_headings_are_accepted(self) -> None:
        cases = {
            "blockquote code": ">     code one\n>     code two\n",
            "bullet code": "- item\n\n      code one\n      code two\n",
            "ordered code": "12. item\n\n        code one\n        code two\n",
            "quoted list code": "> - item\n\n>       code one\n>       code two\n",
            "list quote code": "- > item\n\n  >     code one\n  >     code two\n",
            "list fence": "- ```text\n  wrapped code\n  ```\n",
            "list heading": "- # A structural heading\noutside prose.\n",
            "four-space fence content": "```text\n    ```\nstill code\n```\n",
            "quoted fence": "> ```text\n> wrapped code\n> ```\noutside prose.\n",
            "quoted comment": "> <!--\n> comment body\n> -->\noutside prose.\n",
            "list fence with blank": "- ```text\n  code one\n\n  code two\n  ```\n",
            "empty bullet then code": "-\n\n    code one\n    code two\n",
            "empty ordered item then code": "1.\n\n    code one\n    code two\n",
            "nested empty item then outer code": "- -\n\n      code one\n      code two\n",
            "quoted empty item then quote code": (
                "> -\n\n>     code one\n>     code two\n"
            ),
        }
        for label, source in cases.items():
            with self.subTest(label=label):
                self.assertEqual(markdown_style_violations(source), ())

    def test_list_paragraph_indentation_is_not_mistaken_for_code(self) -> None:
        cases = {
            "bullet second paragraph": (
                "- item\n\n    Second paragraph\n    wrapped.\n",
                4,
            ),
            "ordered second paragraph": (
                "12. item\n\n    Second paragraph\n    wrapped.\n",
                4,
            ),
            "quoted list second paragraph": (
                "> - item\n\n>     Second paragraph\n>     wrapped.\n",
                4,
            ),
            "deep continuation cannot interrupt": (
                "- First half of an item\n      second half.\n",
                2,
            ),
            "empty bullet paragraph": (
                "-\n    First half of an item\n    second half.\n",
                3,
            ),
            "spaced empty bullet paragraph": (
                "-    \n    First half of an item\n    second half.\n",
                3,
            ),
        }
        for label, (source, expected_line) in cases.items():
            with self.subTest(label=label):
                violations = markdown_style_violations(source)
                self.assertTrue(
                    any(item.line == expected_line for item in violations),
                    violations,
                )

    def test_lazy_container_continuations_are_rejected(self) -> None:
        cases = {
            "lazy blockquote": "> First half\nsecond half.\n",
            "partial nested blockquote": "> > First half\n> second half.\n",
            "fully lazy nested blockquote": "> > First half\nsecond half.\n",
            "quoted list missing list prefix": "> - First half\n> second half.\n",
            "quoted list missing all prefixes": "> - First half\nsecond half.\n",
            "lazy list": "- First half\nsecond half.\n",
            "list blockquote explicit": "- > First half\n  > second half.\n",
            "list blockquote missing quote": "- > First half\n  second half.\n",
            "list blockquote missing all prefixes": "- > First half\nsecond half.\n",
        }
        for label, source in cases.items():
            with self.subTest(label=label):
                violations = markdown_style_violations(source)
                self.assertTrue(
                    any(item.line == 2 for item in violations),
                    violations,
                )

        repeated = markdown_style_violations("> First half\nsecond third\nfourth.\n")
        self.assertEqual(
            [item.line for item in repeated if "join this physical line" in item.message],
            [2, 3],
        )

    def test_reference_definitions_do_not_interrupt_open_paragraphs(self) -> None:
        cases = {
            "prose": (
                "First half\n[ref]: /url\nsecond half.\n",
                "prose paragraph",
            ),
            "explicit blockquote": (
                "> First half\n> [ref]: /url\n> second half.\n",
                "blockquote paragraph",
            ),
            "lazy blockquote": (
                "> First half\n[ref]: /url\nsecond half.\n",
                "blockquote paragraph",
            ),
            "list item": (
                "- First half\n  [ref]: /url\n  second half.\n",
                "list item",
            ),
            "quoted list item": (
                "> - First half\n>   [ref]: /url\n>   second half.\n",
                "list item",
            ),
        }
        for label, (source, subject) in cases.items():
            with self.subTest(label=label):
                self.assertEqual(
                    markdown_style_violations(source),
                    (
                        MarkdownViolation(
                            2,
                            f"join this physical line to the preceding {subject}",
                        ),
                        MarkdownViolation(
                            3,
                            f"join this physical line to the preceding {subject}",
                        ),
                    ),
                )

    def test_reference_definitions_at_block_boundaries_are_accepted(self) -> None:
        cases = {
            "document start": "[ref]: /url\nNext paragraph.\n",
            "after blank": "First paragraph.\n\n[ref]: /url\nNext paragraph.\n",
            "blockquote": "> [ref]: /url\n> Quoted prose.\n",
            "list item": "- [ref]: /url\n  Item prose.\n",
        }
        for label, source in cases.items():
            with self.subTest(label=label):
                self.assertEqual(markdown_style_violations(source), ())

    def test_container_leaf_blocks_end_when_their_container_ends(self) -> None:
        cases = {
            "fenced code": "> ```text\noutside first\noutside second.\n",
            "HTML comment": "> <!--\noutside prose\n-->\n",
        }
        for label, source in cases.items():
            with self.subTest(label=label):
                violations = markdown_style_violations(source)
                self.assertTrue(
                    any(
                        item.line == 3 and "join this physical line" in item.message
                        for item in violations
                    ),
                    violations,
                )

    def test_only_commonmark_block_openers_interrupt_paragraphs(self) -> None:
        accepted = {
            "nonempty bullet": "A complete paragraph.\n- A separate item.\n",
            "ordered one": "A complete paragraph.\n1. A separate item.\n",
            "blockquote": "A complete paragraph.\n> A separate quote.\n",
            "heading": "A complete paragraph.\n# A separate heading\n",
            "fence": "A complete paragraph.\n```text\ncode\n```\n",
            "thematic break": "A complete paragraph.\n* * *\nOutside prose.\n",
            "HTML comment": "A complete paragraph.\n<!-- comment -->\nOutside prose.\n",
            "setext underline": "A setext heading\n-\nOutside prose.\n",
            "outside ordered list": "> A complete quote.\n2. An outside list.\n",
            "outside empty item": "> A complete quote.\n-\n",
        }
        for label, source in accepted.items():
            with self.subTest(label=label):
                self.assertEqual(markdown_style_violations(source), ())

        rejected = {
            "ordered two": "First half\n2. second half.\n",
            "empty bullet": "First half\n*\n",
            "deeply indented": "First half\n    second half.\n",
            "deep fence-like continuation": "First half\n    ```\n",
            "blockquote deeply indented": "> First half\n>     second half.\n",
            "ordered two inside blockquote": "> First half\n> 2. second half.\n",
        }
        for label, source in rejected.items():
            with self.subTest(label=label):
                violations = markdown_style_violations(source)
                self.assertTrue(
                    any(item.line == 2 for item in violations),
                    violations,
                )

    def test_completed_containers_do_not_create_lazy_continuations(self) -> None:
        accepted = {
            "quoted blank": "> A complete quote.\n>\nOutside prose.\n",
            "blank line": "> A complete quote.\n\nOutside prose.\n",
            "quoted heading then outside": "> # A quoted heading\nOutside prose.\n",
            "sibling item": "- A complete item.\n- A separate item.\n",
            "list then blockquote": "- A complete item.\n> A separate quote.\n",
        }
        for label, source in accepted.items():
            with self.subTest(label=label):
                self.assertEqual(markdown_style_violations(source), ())

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
