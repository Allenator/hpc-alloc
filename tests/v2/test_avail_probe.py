from __future__ import annotations

import unittest
from types import SimpleNamespace

from hpc_alloc.commands import _partition_gpu_types, _probe_candidates


# sinfo -N -O availability rows: name state host gres gresused cpus
AVAIL = "\n".join(
    (
        "gpu_b200     mix host gpu:b200:8 gpu:b200:4 2/6/0/8",
        "gpu_devel    mix host gpu:b200:8 gpu:b200:0 0/8/0/8",
        "priority_gpu mix host gpu:b200:8 gpu:b200:0 0/8/0/8",
        "gpu_h200     mix host gpu:h200:8 gpu:h200:8 0/8/0/8",
        "day          mix host (null) (null) 2/6/0/8",
    )
)
USER = "GROUPS ab1234 pi_lab01\nASSOC\npi_lab01||interactive,normal\n"
PARTS = "\n".join(
    (
        "PartitionName=gpu_b200 AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=gpu_devel AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=priority_gpu AllowGroups=ALL AllowAccounts=priority State=UP",
        "PartitionName=gpu_h200 AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
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


class FakeClient:
    def user_access(self, netid: str) -> str:
        return USER

    def partition_access(self) -> str:
        return PARTS


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(identity=SimpleNamespace(netid="ab1234")),
        state=FakeState(),
    )


class ProbeCandidateTests(unittest.TestCase):
    def test_partition_gpu_types_parse(self) -> None:
        types = _partition_gpu_types(AVAIL)
        self.assertEqual(types["gpu_b200"], {"b200"})
        self.assertEqual(types["gpu_h200"], {"h200"})
        self.assertEqual(types["day"], set())

    def test_typed_request_filters_by_eligibility_and_gres(self) -> None:
        candidates, capped = _probe_candidates(
            _ctx(), FakeClient(), "bouchet", {"gpus": "b200:1"}, _partition_gpu_types(AVAIL)
        )
        # b200-offering AND eligible only: gpu_b200, gpu_devel.  priority_gpu is
        # excluded (ineligible), gpu_h200 is the wrong type, day has no GPU.
        self.assertEqual(candidates, ["gpu_b200", "gpu_devel"])
        self.assertFalse(capped)

    def test_untyped_gpu_request_matches_any_eligible_gpu_partition(self) -> None:
        candidates, _ = _probe_candidates(
            _ctx(), FakeClient(), "bouchet", {"gpus": "1"}, _partition_gpu_types(AVAIL)
        )
        self.assertEqual(candidates, ["gpu_b200", "gpu_devel", "gpu_h200"])


if __name__ == "__main__":
    unittest.main()
