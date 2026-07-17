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
    REQUEUE_ELIGIBLE_FINAL,
)
from hpc_alloc.models import FinalSource, JobKind, OperationKind, OperationPhase


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


def section(path: Path, heading: str) -> str:
    """Return one Markdown section body, exclusive of later same-or-higher headings.

    Raises rather than returning "" for a missing heading: a check that silently
    searches an empty string passes for every token it is given.
    """

    text = path.read_text()
    level = len(heading) - len(heading.lstrip("#"))
    match = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{path.name}: no section titled {heading!r}")
    rest = text[match.end() :]
    nxt = re.search(rf"^#{{1,{level}}} ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def quoted(body: str) -> set[str]:
    """Every `backticked` token in a section, including those inside code spans."""

    return set(re.findall(r"`([^`]+)`", body))


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


class JsonContractTests(unittest.TestCase):
    """Every key the code emits is named in the contract an agent reads."""

    def setUp(self) -> None:
        self.body = section(CONTRACTS, "## JSON contracts")

    def test_jobs_entry_keys_are_documented(self) -> None:
        emitted = set(payload_keys(_assessment_payload))
        self.assertTrue(emitted, "no keys extracted; the check would pass vacuously")
        undocumented = {k for k in emitted if not mentions(self.body, k)}
        self.assertEqual(
            undocumented,
            set(),
            "status --json emits keys the JSON contract does not name, while that "
            "same contract tells the reader to rely only on documented fields",
        )

    def test_wire_values_are_documented_not_enum_member_names(self) -> None:
        # `final_source` serialises as "confirmed-queue", never "CONFIRMED_QUEUE".
        # An agent matching the prose spelling finds nothing.
        for enum in (FinalSource, JobKind, OperationKind):
            for member in enum:
                with self.subTest(enum=enum.__name__, value=member.value):
                    self.assertTrue(
                        mentions(self.body, member.value),
                        f"{enum.__name__}.{member.name} serialises as "
                        f"{member.value!r}; the JSON contract never names it",
                    )

    def test_every_operation_phase_is_documented(self) -> None:
        # `recover` prints these back verbatim; three went unnamed for a year.
        for member in OperationPhase:
            with self.subTest(phase=member.value):
                self.assertTrue(
                    mentions(self.body, member.value),
                    f"OperationPhase.{member.name} is never named in the JSON contract",
                )

    def test_avail_for_probe_keys_are_documented(self) -> None:
        # The original drift: the probe object gained `preemptible` and the
        # documented shape did not.  `preemptible` is discussed elsewhere in
        # the file, so only a section-scoped check catches it.
        literals = dict_literals_containing(COMMANDS, "schedulable")
        self.assertTrue(literals, "no probe dict literal found in commands.py")
        for key in set().union(*literals):
            with self.subTest(key=key):
                self.assertTrue(
                    mentions(self.body, key),
                    f"avail --for probes carry {key!r}; the JSON contract omits it",
                )

    def test_avail_for_envelope_keys_are_documented(self) -> None:
        # Covers both the success payload and the unknown-GPU-type error one,
        # which still carries `capped`.
        literals = dict_literals_containing(COMMANDS, "probes")
        self.assertTrue(literals, "no avail --for payload literal found")
        for key in set().union(*literals):
            with self.subTest(key=key):
                self.assertTrue(
                    mentions(self.body, key),
                    f"avail --for --json carries {key!r}; the JSON contract omits it",
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
