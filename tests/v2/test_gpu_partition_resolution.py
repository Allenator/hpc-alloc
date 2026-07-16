from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace

from hpc_alloc.commands import _is_dedicated_gpu_partition, _resolve_gpu_partition
from hpc_alloc.errors import ConfigInvalid, TransportLost


# `sinfo -h -N -O Partition,StateCompact,NodeHost,Gres,GresUsed,CPUsState` rows,
# modelled on Bouchet: the static `gpu` default hosts only rtx_5000_ada, and each
# other GPU type has its own dedicated partition, plus the preemptible `scavenge*`
# pool, the short `gpu_devel` pool, and the account-gated `priority_gpu`.
AVAIL = "\n".join(
    (
        "gpu          mix host gpu:rtx_5000_ada:4 gpu:rtx_5000_ada:0 2/6/0/8",
        "gpu_h200     mix host gpu:h200:8 gpu:h200:0 0/8/0/8",
        "gpu_b200     mix host gpu:b200:8 gpu:b200:0 0/8/0/8",
        "gpu_rtx6000  mix host gpu:rtx_pro_6000_blackwell:8 gpu:rtx_pro_6000_blackwell:0 0/8/0/8",
        "gpu_devel    mix host gpu:h200:8 gpu:h200:0 0/8/0/8",
        "priority_gpu mix host gpu:h200:8 gpu:h200:0 0/8/0/8",
        "scavenge_gpu mix host gpu:h200:8 gpu:h200:0 0/8/0/8",
        "scavenge     mix host gpu:h200:8 gpu:h200:0 0/8/0/8",
        "day          mix host (null) (null) 2/6/0/8",
    )
)
# ab1234 belongs to account pi_lab01 with QOS interactive,normal -- not the
# `priority` account priority_gpu demands, nor a `scavenge` QOS.
USER = "GROUPS ab1234 pi_lab01\nASSOC\npi_lab01||interactive,normal\n"
PARTS = "\n".join(
    (
        "PartitionName=gpu AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=gpu_h200 AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=gpu_b200 AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=gpu_rtx6000 AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=gpu_devel AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
        "PartitionName=priority_gpu AllowGroups=ALL AllowAccounts=priority State=UP",
        "PartitionName=scavenge_gpu AllowGroups=ALL AllowAccounts=ALL AllowQos=scavenge State=UP",
        "PartitionName=scavenge AllowGroups=ALL AllowAccounts=ALL AllowQos=scavenge State=UP",
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
    def __init__(
        self,
        *,
        availability: str = AVAIL,
        availability_error: Exception | None = None,
    ) -> None:
        self._availability = availability
        self._availability_error = availability_error

    def availability(self) -> str:
        if self._availability_error is not None:
            raise self._availability_error
        return self._availability

    def user_access(self, netid: str) -> str:
        return USER

    def partition_access(self) -> str:
        return PARTS


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(identity=SimpleNamespace(netid="ab1234")),
        state=FakeState(),
    )


def _resources(gpus, partition="gpu") -> dict:
    return {"partition": partition, "gpus": gpus}


class DedicatedPartitionTests(unittest.TestCase):
    def test_scavenge_and_devel_are_not_dedicated(self) -> None:
        self.assertFalse(_is_dedicated_gpu_partition("scavenge"))
        self.assertFalse(_is_dedicated_gpu_partition("scavenge_gpu"))
        self.assertFalse(_is_dedicated_gpu_partition("gpu_devel"))
        self.assertTrue(_is_dedicated_gpu_partition("gpu_h200"))
        self.assertTrue(_is_dedicated_gpu_partition("gpu"))


class ResolveGpuPartitionTests(unittest.TestCase):
    def _resolve(self, resources, client=None):
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            result = _resolve_gpu_partition(
                _ctx(), client or FakeClient(), "bouchet", resources
            )
        return result, buffer.getvalue()

    def test_unique_dedicated_eligible_partition_is_auto_selected(self) -> None:
        # h200 is offered by gpu_h200 (dedicated/eligible), gpu_devel (short),
        # priority_gpu (wrong account), and the scavenge* pool (preemptible).
        # Only gpu_h200 survives, so it is chosen and announced.
        resources = _resources("h200:1")
        chosen, stderr = self._resolve(resources)
        self.assertEqual(chosen, "gpu_h200")
        self.assertEqual(resources["partition"], "gpu")  # helper does not mutate
        self.assertIn("gpu_h200", stderr)
        self.assertIn("h200:1", stderr)

    def test_default_partition_that_offers_type_is_kept(self) -> None:
        # gpu genuinely offers rtx_5000_ada, so the static default is unchanged.
        chosen, _ = self._resolve(_resources("rtx_5000_ada:1"))
        self.assertEqual(chosen, "gpu")

    def test_bare_count_keeps_static_default_without_a_fetch(self) -> None:
        # No type to steer by; a client whose availability() would raise proves
        # the topology is never fetched for a bare count.
        client = FakeClient(availability_error=AssertionError("must not fetch"))
        chosen, _ = self._resolve(_resources("2"), client=client)
        self.assertEqual(chosen, "gpu")

    def test_no_partition_offers_type_is_refused(self) -> None:
        with self.assertRaisesRegex(
            ConfigInvalid, r"no partition offers notagpu GPUs on bouchet"
        ):
            self._resolve(_resources("notagpu:1"))

    def test_type_only_on_preemptible_or_short_partitions_is_refused(self) -> None:
        # l40s exists only on scavenge_gpu and gpu_devel here: no steady, eligible
        # partition offers it, so the tool refuses rather than auto-picking a
        # preemptible or short pool.
        avail = AVAIL + "\nscavenge_gpu mix host gpu:l40s:4 gpu:l40s:0 0/4/0/4"
        avail += "\ngpu_devel    mix host gpu:l40s:4 gpu:l40s:0 0/4/0/4"
        client = FakeClient(availability=avail)
        with self.assertRaisesRegex(
            ConfigInvalid, r"no partition offers l40s GPUs on bouchet"
        ):
            self._resolve(_resources("l40s:1"), client=client)

    def test_single_ineligible_dedicated_partition_is_still_selected(self) -> None:
        # h200x is offered by exactly one dedicated partition, priority_gpu, which
        # the user is ineligible for.  Eligibility is a tie-breaker, not a gate: a
        # lone candidate is still auto-selected so the authoritative submit can
        # reject the access error cleanly, instead of refusing here on a mirror.
        avail = "priority_gpu mix host gpu:h200x:8 gpu:h200x:0 0/8/0/8"
        chosen, stderr = self._resolve(
            _resources("h200x:1"), client=FakeClient(availability=avail)
        )
        self.assertEqual(chosen, "priority_gpu")
        self.assertIn("priority_gpu", stderr)

    def test_multiple_dedicated_eligible_partitions_are_refused(self) -> None:
        # Two steady, eligible partitions offer b200: refuse and list both.
        avail = AVAIL + "\ngpu_b200x    mix host gpu:b200:8 gpu:b200:0 0/8/0/8"
        parts = PARTS + (
            "\nPartitionName=gpu_b200x AllowGroups=ALL AllowAccounts=ALL "
            "AllowQos=normal State=UP"
        )

        class TwoB200Client(FakeClient):
            def partition_access(self) -> str:
                return parts

        client = TwoB200Client(availability=avail)
        with self.assertRaisesRegex(
            ConfigInvalid, r"multiple partitions offer b200.*gpu_b200, gpu_b200x"
        ):
            self._resolve(_resources("b200:1"), client=client)

    def test_availability_failure_falls_open_to_static_default(self) -> None:
        client = FakeClient(availability_error=TransportLost("VPN down"))
        chosen, stderr = self._resolve(_resources("h200:1"), client=client)
        self.assertEqual(chosen, "gpu")
        self.assertIn("gpu", stderr)

    def test_bare_default_without_gpus_is_unchanged(self) -> None:
        chosen, _ = self._resolve({"partition": "day", "gpus": None})
        self.assertEqual(chosen, "day")


if __name__ == "__main__":
    unittest.main()
