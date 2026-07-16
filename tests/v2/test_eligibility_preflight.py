from __future__ import annotations

import unittest
from types import SimpleNamespace

from hpc_alloc.commands import _eligibility_snapshot, _preflight_partition_eligibility
from hpc_alloc.errors import ConfigInvalid, TransportLost


USER = "GROUPS ab1234 pi_lab01\nASSOC\npi_lab01||interactive,normal\n"
PARTS = "\n".join(
    (
        "PartitionName=priority_gpu AllowGroups=ALL AllowAccounts=priority "
        "DenyQos=normal,interactive State=UP",
        "PartitionName=gpu_b200 AllowGroups=ALL AllowAccounts=ALL "
        "AllowQos=normal,nothrottle State=UP",
    )
)


class FakeState:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], object] = {}
        self.sets = 0

    def get_cluster_cache(self, cluster: str, key: str) -> object:
        return self.store.get((cluster, key))

    def set_cluster_cache(self, cluster, key, value, *, expires_at=None) -> None:
        self.sets += 1
        self.store[(cluster, key)] = value


class FakeClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def user_access(self, netid: str) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return USER

    def partition_access(self) -> str:
        if self.error:
            raise self.error
        return PARTS


def _ctx(state: FakeState) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(identity=SimpleNamespace(netid="ab1234")),
        state=state,
    )


class PreflightEligibilityTests(unittest.TestCase):
    def test_ineligible_partition_is_refused_locally(self) -> None:
        with self.assertRaisesRegex(ConfigInvalid, r"priority_gpu.*no submission attempted"):
            _preflight_partition_eligibility(
                _ctx(FakeState()), FakeClient(), "bouchet", "priority_gpu"
            )

    def test_eligible_partition_passes(self) -> None:
        _preflight_partition_eligibility(
            _ctx(FakeState()), FakeClient(), "bouchet", "gpu_b200"
        )

    def test_unknown_partition_falls_open(self) -> None:
        _preflight_partition_eligibility(
            _ctx(FakeState()), FakeClient(), "bouchet", "not_a_partition"
        )

    def test_fetch_failure_falls_open(self) -> None:
        # A transport failure must not block a submit -- fall open, no raise.
        _preflight_partition_eligibility(
            _ctx(FakeState()), FakeClient(error=TransportLost("VPN down")), "bouchet", "priority_gpu"
        )

    def test_snapshot_is_cached_after_first_fetch(self) -> None:
        state = FakeState()
        ctx = _ctx(state)
        client = FakeClient()
        first = _eligibility_snapshot(ctx, client, "bouchet")
        second = _eligibility_snapshot(ctx, client, "bouchet")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(client.calls, 1)  # second snapshot served from cache
        self.assertEqual(state.sets, 1)


if __name__ == "__main__":
    unittest.main()
