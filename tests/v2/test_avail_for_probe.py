from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from hpc_alloc.commands import _avail_probe


# sinfo -h -N -O availability rows: name state host gres gresused cpus.
AVAIL = "\n".join(
    (
        "gpu_b200     mix host gpu:b200:8 gpu:b200:4 2/6/0/8",   # free b200 = 4
        "gpu_devel    mix host gpu:b200:8 gpu:b200:0 0/8/0/8",   # free = 8, *devel -> preemptible
        "priority_gpu mix host gpu:b200:8 gpu:b200:0 0/8/0/8",   # priority account -> ineligible
        "day          mix host (null) (null) 2/6/0/8",
    )
)
USER = "GROUPS ab1234 pi_lab01\nASSOC\npi_lab01||interactive,normal\n"
PARTS = "\n".join(
    (
        "PartitionName=gpu_b200 AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=gpu_devel AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=priority_gpu AllowGroups=ALL AllowAccounts=priority State=UP",
        "PartitionName=day AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
    )
)


class FakeState:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], object] = {}

    def get_cluster_cache(self, cluster: str, key: str) -> object:
        return self.store.get((cluster, key))

    def set_cluster_cache(self, cluster, key, value, *, expires_at=None) -> None:
        self.store[(cluster, key)] = value


class ProbeClient:
    def __init__(self) -> None:
        self.probed: list[str] = []

    def availability(self) -> str:
        return AVAIL

    def user_access(self, netid: str) -> str:
        return USER

    def partition_access(self) -> str:
        return PARTS

    def schedule_probe(self, *, partition, walltime, cpus, mem=None, gpus=None, constraint=None) -> str:
        self.probed.append(partition)
        return (
            f"sbatch: Job 1 to start at 2026-07-16T12:00:00 using {cpus} "
            f"processors in partition {partition}"
        )


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            resolve_option=lambda option, cluster, fallback=None: fallback,
        ),
        state=FakeState(),
    )


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        json=True, partition=None, gpus=None, cpus=2, time="1:00:00",
        mem=None, constraint=None, probe=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class AvailForProbeTests(unittest.TestCase):
    def _run(self, client, **overrides) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = _avail_probe(_args(**overrides), _ctx(), client, "bouchet")
        self.assertEqual(code, 0)
        return buffer.getvalue()

    def test_json_orders_by_capacity_flags_preemptible_and_reports_capped(self) -> None:
        payload = json.loads(self._run(ProbeClient(), gpus="b200:1"))
        self.assertFalse(payload["capped"])  # only two candidates
        parts = [p["partition"] for p in payload["probes"]]
        # ordered by free b200 capacity -- gpu_devel (8) before gpu_b200 (4).
        self.assertEqual(parts, ["gpu_devel", "gpu_b200"])
        flags = {p["partition"]: p["preemptible"] for p in payload["probes"]}
        self.assertTrue(flags["gpu_devel"])   # *devel flagged preemptible
        self.assertFalse(flags["gpu_b200"])

    def test_explicit_ineligible_partition_is_reported_not_raised(self) -> None:
        # a read-only probe reports the partition instead of raising.
        client = ProbeClient()
        payload = json.loads(self._run(client, partition="priority_gpu"))
        self.assertEqual([p["partition"] for p in payload["probes"]], ["priority_gpu"])
        self.assertEqual(client.probed, ["priority_gpu"])

    def test_text_header_includes_constraint_and_marks_preemptible(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = _avail_probe(
                _args(gpus="b200:1", constraint="bigmem", json=False),
                _ctx(), ProbeClient(), "bouchet",
            )
        self.assertEqual(code, 0)
        self.assertIn("-C bigmem", err.getvalue())  # header via info() -> stderr
        self.assertIn("(preemptible)", out.getvalue())  # marked in the stdout table

    def test_unknown_gpu_type_reports_capped_false(self) -> None:
        payload = json.loads(self._run(ProbeClient(), gpus="notagpu:1"))
        self.assertIn("unknown or unavailable GPU type", payload["error"])
        self.assertFalse(payload["capped"])
        self.assertEqual(payload["probes"], [])


if __name__ == "__main__":
    unittest.main()
