"""Pin the documented contract to the code it describes.

The suite verifies the code.  Nothing verified the claims *about* the code, so
the documentation could -- and did -- assert the opposite of what the software
does: an audit of 277 documented claims found a JSON payload missing an emitted
field, three operation phases `recover` prints but no document named, and
`final_source` spelled in the enum's member names rather than the wire values it
serialises to.  Every one of those is a set-equality question, so every one of
them is testable.

What this file can and cannot do:

* It pins ENUMERABLE facts -- field names, enum values, scheduler-state sets,
  flag names.  Those are the drift a reader cannot detect by reading.
* It cannot pin PROSE.  "A live exact match resolves the operation" is a claim
  about control flow, and the worst defect the audit found lived exactly there.
  Fewer copies of the prose is the only lever for that; see the README/reference
  split.

Checks are SECTION-SCOPED on purpose.  A file-wide substring search passes as
soon as a token appears anywhere -- `preemptible` is discussed elsewhere in both
documents, so a file-wide check would have called the JSON contract correct while
it omitted that very field.  The section is the unit that has to be right.

The direction is COMPLETENESS: everything the code has, the documents must name.
The reverse (a document naming something the code lacks) is not checked here --
prose legitimately mentions Slurm states and concepts that are not our enums, so
an exactness check would drown in false positives.

The unit is the section, not the shape diagram.  These checks prove a section
names every field the code emits; they do not prove the illustrative
`{ "probes": [ ... ] }` sketches are exhaustive, because pinning those would
couple the tests to one way of writing them.  Dropping a field from a sketch
while its prose survives therefore passes -- the reader is still told the field
exists, which is the property worth defending.
"""

from __future__ import annotations

import argparse
import ast
import re
import tempfile
import unittest
from pathlib import Path

from hpc_alloc.cli import build_parser
from hpc_alloc.commands import _assessment_payload
from hpc_alloc.lifecycle import (
    _ACTIVE,
    _CANCELLATION_DRAINING,
    _QUEUED,
    _REQUEUEING,
    _STARTED_INACTIVE,
    AssessmentPhase,
    REQUEUE_ELIGIBLE_FINAL,
)
from hpc_alloc.models import (
    FinalSource,
    JobKind,
    JobPhase,
    OperationKind,
    OperationPhase,
)


REPO = Path(__file__).resolve().parents[2]
CONTRACTS = REPO / "skill" / "references" / "command-contracts.md"
LIFECYCLE = REPO / "skill" / "references" / "recovery-and-lifecycle.md"
README = REPO / "README.md"
COMMANDS = REPO / "hpc_alloc" / "commands.py"


def dict_literals_containing(path: Path, key: str) -> list[list[str]]:
    """Key lists of every dict literal in *path* that carries *key*.

    Some payloads are built inline inside a `json.dumps` call rather than by a
    named function, so there is nothing to import and call.  Reading the literal
    is still exact, and finding none raises at the call site rather than
    quietly checking an empty set.
    """

    literals = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            k.value
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
        if key in keys:
            literals.append(keys)
    return literals


# Leading whitespace is allowed: CommonMark permits up to three spaces before a
# fence, and a fence nested in a list item is indented further -- which is the
# most natural way for someone to add a code sample to a field description.  An
# unclosed fence runs to end of input rather than being ignored, so a malformed
# document fails loudly instead of silently reverting to the old behaviour.
_FENCE = re.compile(r"^[ \t]*```.*?(?:^[ \t]*```|\Z)", re.MULTILINE | re.DOTALL)


def without_fences(text: str) -> str:
    """Blank out fenced code blocks, preserving line count.

    Fences break both helpers below.  A ``` fence reads to `quoted` as one
    enormous code span, so every token inside it would count as documented --
    a sample containing `SUBMITTING` would satisfy the check that the prose
    documents `SUBMITTING`.  And a shell comment inside a fence ("# Inspect
    ...") matches the heading pattern `section` scans for, truncating the
    section at it.  Neither bites today, because no checked section contains a
    fence; nothing stops one being added, and the first failure mode is silent.

    FenceHandlingTests pins this.  Without them the helper is invisible:
    neutering it to `return text` left every other test in this file green.
    """

    return _FENCE.sub(lambda m: "\n" * m.group().count("\n"), text)


def section(path: Path, heading: str) -> str:
    """Return one Markdown section body, exclusive of later same-or-higher headings.

    Raises rather than returning "" for a missing heading: a check that silently
    searches an empty string passes for every token it is given.
    """

    text = without_fences(path.read_text())
    level = len(heading) - len(heading.lstrip("#"))
    match = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{path.name}: no section titled {heading!r}")
    rest = text[match.end() :]
    nxt = re.search(rf"^#{{1,{level}}} ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def quoted(body: str) -> set[str]:
    """Every `backticked` token in a section, excluding fenced code blocks.

    Safe on raw text as well as on a `section` result, which strips fences
    already; the second pass is what makes this usable on either.
    """

    return set(re.findall(r"`([^`\n]+)`", without_fences(body)))


def mentions(body: str, token: str) -> bool:
    """True when `token` appears as its own backticked span, or inside one.

    A flag documented as `hpc-alloc up [--name N]` carries `--name` inside a
    larger span, so exact-span matching alone would miss it.  The boundary
    excludes `-` and `_` as well as word characters, so `time` does not match
    inside `--time` and `kind` does not match inside `job_kind`.
    """

    pattern = re.compile(rf"(?<![\w-]){re.escape(token)}(?![\w-])")
    return any(token == span or pattern.search(span) for span in quoted(body))


def payload_keys(function: object) -> list[str]:
    """Static keys of a function whose body is one `return {...}` literal.

    Read with ast rather than by calling it: a fixture standing in for a
    JobRecord would have to satisfy every attribute the literal touches, and
    would drift from the real shape without anyone noticing.  If the function
    stops being a single dict literal this raises, which is the correct
    outcome -- the check must fail loudly rather than quietly stop checking.
    """

    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = tree.body[0].body  # type: ignore[attr-defined]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        raise AssertionError("expected a single return statement")
    literal = body[0].value
    if not isinstance(literal, ast.Dict):
        raise AssertionError("expected the return value to be a dict literal")
    keys = [k.value for k in literal.keys if isinstance(k, ast.Constant)]
    if len(keys) != len(literal.keys):
        raise AssertionError("dict literal has computed keys; cannot read statically")
    return keys


class FenceHandlingTests(unittest.TestCase):
    """A code sample must not stand in for documentation.

    These pin the PROPERTY, not any one mechanism, because two independent
    mechanisms hold it: `without_fences` blanks the block, and `quoted`'s
    newline exclusion stops a multi-line span forming in the first place.
    Either alone is sufficient, so neutering one leaves the token checks below
    green -- only breaking both fails them.  That redundancy is deliberate but
    worth stating, because it means these tests do NOT guard `without_fences`
    on `quoted`'s path.

    What `without_fences` uniquely carries is `section`: a shell comment inside
    a fence matches the heading pattern, and only stripping the block prevents
    it truncating a section.  test_a_comment_inside_a_fence_does_not_end_a_section
    is the one check that fails when the helper is removed.

    They exist because every other test in this file used the helper and none
    exercised it: a helper whose removal no test notices is not a guard, it is
    a comment.
    """

    def test_a_fenced_sample_does_not_document_a_token(self) -> None:
        fenced = "Prose that explains nothing.\n\n```text\nSUBMITTING\n```\n"
        self.assertFalse(
            mentions(fenced, "SUBMITTING"),
            "a token inside a code sample counted as documented",
        )
        self.assertTrue(
            mentions("Prose naming `SUBMITTING` explicitly.", "SUBMITTING"),
            "sanity: real prose must still count, or the check is vacuous",
        )

    def test_an_indented_fence_is_still_stripped(self) -> None:
        # A sample nested in a list item is the likeliest way one arrives.
        nested = "- a field:\n\n  ```text\n  SUBMITTING\n  ```\n"
        self.assertFalse(mentions(nested, "SUBMITTING"))

    def test_an_unclosed_fence_does_not_silently_revert(self) -> None:
        # Runs to end of input rather than being skipped, so a malformed
        # document cannot quietly restore the pre-fix behaviour.
        self.assertFalse(mentions("text\n\n```text\nSUBMITTING\n", "SUBMITTING"))

    def test_a_comment_inside_a_fence_does_not_end_a_section(self) -> None:
        body = (
            "## Target\n\nprose `EARLY` here\n\n"
            "```bash\n# Inspect something\necho hi\n```\n\n"
            "prose `LATE` here\n\n## Next\n\nnot in the section\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "doc.md"
            path.write_text(body)
            extracted = section(path, "## Target")
        self.assertIn("`EARLY`", extracted)
        self.assertIn("`LATE`", extracted, "a '#' comment in a fence ended the section")
        self.assertNotIn("not in the section", extracted)


class JsonContractTests(unittest.TestCase):
    """Every key the code emits is named wherever the JSON surface is documented.

    Both copies are checked.  The JSON surface is the one contract a script
    depends on, and the original drift -- a probe object that gained
    `preemptible` while its documented shape did not -- was in the README copy
    precisely because nothing tested it.  Duplication is fine here exactly
    because it is pinned; see the module docstring.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Every section that documents the JSON surface, each held to the same
        # bar, read once rather than per token.
        cls.bodies = (
            ("command-contracts.md", section(CONTRACTS, "## JSON contracts")),
            ("README.md", section(README, "## JSON contracts")),
        )

    def assert_documented(self, token: str, message: str) -> None:
        for name, body in self.bodies:
            with self.subTest(document=name):
                self.assertTrue(mentions(body, token), f"{name}: {message}")

    def test_jobs_entry_keys_are_documented(self) -> None:
        emitted = set(payload_keys(_assessment_payload))
        self.assertTrue(emitted, "no keys extracted; the check would pass vacuously")
        for key in sorted(emitted):
            self.assert_documented(
                key,
                f"status --json emits {key!r}, which the JSON contract does not "
                f"name, while telling the reader to rely only on documented fields",
            )

    def test_wire_values_are_documented_not_enum_member_names(self) -> None:
        # `final_source` serialises as "confirmed-queue", never "CONFIRMED_QUEUE".
        # An agent matching the prose spelling finds nothing.
        for enum in (FinalSource, JobKind, OperationKind):
            for member in enum:
                self.assert_documented(
                    member.value,
                    f"{enum.__name__}.{member.name} serialises as "
                    f"{member.value!r}; the JSON contract never names it",
                )

    def test_every_operation_phase_is_documented(self) -> None:
        # `recover` prints these back verbatim; three went unnamed for a year.
        for member in OperationPhase:
            self.assert_documented(
                member.value,
                f"OperationPhase.{member.name} is never named in the JSON contract",
            )

    def test_every_job_phase_value_is_documented(self) -> None:
        """`jobs[].phase` draws on two enums and a literal, so pin the union.

        commands.py:1269 emits an AssessmentPhase; :1333 emits the durable
        JobPhase when a job has no acknowledged Slurm ID, which is how
        `SUBMITTING` reaches the wire; :1338 and :1377 force the literal
        "UNCERTAIN" for a cluster that could not be scanned.  An earlier draft
        of this file pinned only the enums it happened to import, and
        `SUBMITTING` was consequently absent from both contracts -- the one
        value that must never be read as "nothing was submitted".
        """

        domain = {m.value for m in AssessmentPhase} | {m.value for m in JobPhase}
        self.assertIn("SUBMITTING", domain, "sanity: the durable phase reaches jobs[]")
        self.assertIn("UNCERTAIN", domain, "sanity: the forced literal is in the union")
        for value in sorted(domain):
            self.assert_documented(
                value,
                f"jobs[].phase can be {value!r}; the JSON contract never names it",
            )

    def test_avail_for_probe_keys_are_documented(self) -> None:
        # The original drift: the probe object gained `preemptible` and the
        # documented shape did not.  `preemptible` is discussed elsewhere in
        # the file, so only a section-scoped check catches it.
        literals = dict_literals_containing(COMMANDS, "schedulable")
        self.assertTrue(literals, "no probe dict literal found in commands.py")
        for key in sorted(set().union(*literals)):
            self.assert_documented(
                key, f"avail --for probes carry {key!r}; the JSON contract omits it"
            )

    def test_avail_for_envelope_keys_are_documented(self) -> None:
        # Covers both the success payload and the unknown-GPU-type error one,
        # which still carries `capped`.
        literals = dict_literals_containing(COMMANDS, "probes")
        self.assertTrue(literals, "no avail --for payload literal found")
        for key in sorted(set().union(*literals)):
            self.assert_documented(
                key, f"avail --for --json carries {key!r}; the JSON contract omits it"
            )


class LifecycleContractTests(unittest.TestCase):
    """The scheduler-state taxonomy the reference spells out matches the code.

    Scoped per section, not per file.  Every one of these state names appears
    somewhere in this reference, so a file-wide check would confirm nothing:
    delete `STAGE_OUT` from the kill sequence and it still occurs in the state
    map two sections later.
    """

    def setUp(self) -> None:
        self.body = section(LIFECYCLE, "## Interpret lifecycle evidence")

    def test_every_mapped_scheduler_state_is_documented(self) -> None:
        for label, states in (
            ("ACTIVE", _ACTIVE),
            ("QUEUED", _QUEUED),
            ("STARTED_INACTIVE", _STARTED_INACTIVE),
            ("REQUEUEING", _REQUEUEING),
            ("requeue-eligible final", REQUEUE_ELIGIBLE_FINAL),
        ):
            for state in states:
                with self.subTest(group=label, state=state):
                    self.assertIn(
                        f"`{state}`",
                        self.body,
                        f"{state} maps to {label} but the reference never names it",
                    )

    def test_cancellation_draining_states_are_documented(self) -> None:
        # These three decide whether an ambiguous cancellation resolves or
        # stays unresolved -- the single most consequential paragraph in the
        # reference, and one whose prose no test can check.  The set at least
        # cannot drift away from it silently.
        body = section(LIFECYCLE, "## Reconcile cancellation uncertainty")
        self.assertTrue(_CANCELLATION_DRAINING, "empty set would pass vacuously")
        for state in _CANCELLATION_DRAINING:
            with self.subTest(state=state):
                self.assertIn(
                    f"`{state}`",
                    body,
                    f"{state} is part of the kill sequence that leaves a "
                    f"cancellation unresolved, but the section that decides "
                    f"resolution never names it",
                )


class CommandSurfaceTests(unittest.TestCase):
    """Every subcommand and long flag reaches both documents.

    Short flags are not checked: `-c` and `-n` are single characters that match
    almost any prose, so a containment check on them proves nothing.
    """

    def setUp(self) -> None:
        parser = build_parser()
        (self.subparsers,) = [
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        ]
        self.contracts = section(CONTRACTS, "## Command surface")
        self.readme = section(README, "## Commands")

    def test_every_subcommand_is_documented(self) -> None:
        for name in self.subparsers.choices:
            with self.subTest(command=name):
                self.assertTrue(
                    mentions(self.contracts, f"hpc-alloc {name}"),
                    f"{name} is a subcommand the contract table omits",
                )
                self.assertTrue(
                    mentions(self.readme, name),
                    f"{name} is a subcommand the README's command table omits",
                )

    def test_every_long_flag_is_documented(self) -> None:
        for command, sub in self.subparsers.choices.items():
            for action in sub._actions:
                for flag in action.option_strings:
                    if not flag.startswith("--") or flag == "--help":
                        continue
                    with self.subTest(command=command, flag=flag):
                        self.assertTrue(
                            mentions(self.contracts, flag),
                            f"{command} accepts {flag}; the contract table omits it",
                        )
                        self.assertTrue(
                            mentions(self.readme, flag),
                            f"{command} accepts {flag}; the README table omits it",
                        )


if __name__ == "__main__":
    unittest.main()
