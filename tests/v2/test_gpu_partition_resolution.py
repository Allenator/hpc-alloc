from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from hpc_alloc.commands import (
    _TOPOLOGY_CACHE_KEY,
    _eligibility_snapshot,
    _is_dedicated_gpu_partition,
    _resolve_gpu_partition,
    _topology_snapshot,
)
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


class TopologyCacheTests(unittest.TestCase):
    def test_topology_snapshot_caches_then_serves_offline(self) -> None:
        ctx = _ctx()
        first = _topology_snapshot(ctx, FakeClient(), "bouchet")
        assert first is not None
        self.assertEqual(first["gpu_h200"], {"h200"})
        # A warm cluster is served offline (no client); a cold one yields None.
        self.assertEqual(_topology_snapshot(ctx, None, "bouchet", fetch=False), first)
        self.assertIsNone(_topology_snapshot(ctx, None, "elsewhere", fetch=False))

    def test_a_row_it_cannot_read_in_full_is_a_miss_not_an_empty_cluster(self) -> None:
        """An unreadable cache row must defer, never describe the cluster.

        Filtering per entry dropped what it could not read: an all-bad row
        became {}, which is not None and so read downstream as "no partition
        offers this GPU" -- blaming the user's spelling for a bad cache.  A
        partly-bad row was worse: a partial map the resolver auto-selected from
        with no error.  The element check is not redundant; the row is JSON, so
        [1, 2] is a list whose set matches no GPU type.
        """

        for label, row in (
            ("all values unreadable", {"gpu_h200": "h200"}),
            ("one value unreadable", {"gpu_h200": ["h200"], "priority_gpu": "h200"}),
            ("non-str elements", {"gpu_h200": [1, 2]}),
        ):
            with self.subTest(row=label):
                ctx = _ctx()
                ctx.state.set_cluster_cache("bouchet", _TOPOLOGY_CACHE_KEY, row)
                self.assertIsNone(
                    _topology_snapshot(ctx, None, "bouchet", fetch=False),
                    "an unreadable row was reported as cluster topology",
                )
                # ...and a fetch overwrites it rather than stranding it to TTL.
                self.assertEqual(
                    _topology_snapshot(ctx, FakeClient(), "bouchet")["gpu_h200"],
                    {"h200"},
                )

    def test_offline_resolve_defers_when_cold_and_selects_when_warm(self) -> None:
        # This is the "warm eagerly on connect" story: cold, the offline resolve
        # defers (None); after both caches are warmed (as connect does), the same
        # offline resolve picks the eligible partition with no client at all.
        ctx = _ctx()
        self.assertIsNone(
            _resolve_gpu_partition(ctx, None, "bouchet", _resources("h200:1"), fetch=False)
        )
        _topology_snapshot(ctx, FakeClient(), "bouchet")
        _eligibility_snapshot(ctx, FakeClient(), "bouchet")
        chosen = _resolve_gpu_partition(
            ctx, None, "bouchet", _resources("h200:1"), fetch=False
        )
        self.assertEqual(chosen, "gpu_h200")


class DryRunOfflineResolveTests(unittest.TestCase):
    def _dry_run(
        self,
        *,
        warm: bool,
        warm_eligibility: bool = True,
        client: object | None = None,
        gpus: str = "h200:1",
    ):
        from hpc_alloc.commands import _submit_job
        from hpc_alloc.models import JobKind

        client = client or FakeClient()
        ctx = _ctx()
        if warm:  # as `connect` does: warm both accelerator caches
            _topology_snapshot(ctx, client, "bouchet")
            if warm_eligibility:
                _eligibility_snapshot(ctx, client, "bouchet")
        resources = {
            "partition": "gpu",
            "gpus": gpus,
            "time": "1:00:00",
            "cpus": 2,
            "mem": None,
            "constraint": None,
            "chdir": None,
        }
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            _submit_job(
                ctx=ctx,
                paths=None,
                entrypoint=None,
                cluster="bouchet",
                kind=JobKind.ALLOCATION,
                logical_name="dev",
                resources=resources,
                wrap="sleep infinity",
                logfile_template=".hpc-alloc/alloc-{operation_id}.log",
                dry_run=True,
            )
        return out.getvalue(), err.getvalue()

    def test_warm_cache_prints_the_resolved_partition(self) -> None:
        # with a warm topology+access cache the dry-run command matches what a
        # real submit would do -- the resolved partition, not the static default.
        out, err = self._dry_run(warm=True)
        self.assertIn("--partition=gpu_h200", out)
        self.assertNotIn("stays offline", err)

    def test_cold_cache_warns_and_keeps_the_static_default(self) -> None:
        out, err = self._dry_run(warm=False)
        self.assertIn("--partition=gpu", out)
        self.assertNotIn("--partition=gpu_h200", out)
        self.assertIn("stays offline", err)

    def test_cold_access_rules_defer_rather_than_invent_an_ambiguity(self) -> None:
        """Topology warm, access rules cold: defer; never refuse.

        Eligibility is the tie-breaker that narrows the account-gated
        `priority_gpu` away, so without it `allowed` degenerates to `dedicated`
        and a pick the real submit makes silently looks ambiguous.  Reporting
        that as a refusal contradicts the submit this previews and recommends a
        partition the account is denied.  An earlier version of this test
        asserted that refusal was correct.

        This is the check that fails if the deferral is reverted, so it asserts
        the deferral is PRESENT and the refusal ABSENT -- either alone would
        pass against the wrong behaviour.
        """

        out, err = self._dry_run(warm=True, warm_eligibility=False)
        self.assertIn("--partition=gpu", out)  # always-prints contract holds
        self.assertNotIn("--partition=gpu_h200", out)
        self.assertIn("cannot resolve a partition", err)
        self.assertNotIn("multiple partitions offer", err)
        self.assertNotIn("no cached topology", err)

    def test_the_state_that_defers_is_the_state_the_submit_resolves(self) -> None:
        """Pin the pair, not each side: deferring is only right because the
        submit picks.  Asserting the dry-run alone would still pass if the
        resolver deferred for a bad reason."""

        ctx = _ctx()
        _topology_snapshot(ctx, FakeClient(), "bouchet")  # eligibility left cold
        self.assertIsNone(
            _resolve_gpu_partition(ctx, None, "bouchet", _resources("h200:1"), fetch=False)
        )
        self.assertEqual(
            _resolve_gpu_partition(
                ctx, FakeClient(), "bouchet", _resources("h200:1"), fetch=True
            ),
            "gpu_h200",
        )

    def test_a_genuine_ambiguity_on_warm_caches_is_still_reported(self) -> None:
        """The deferral must not swallow a real refusal.

        Two dedicated partitions the account can actually use offer b200, so
        the tie-breaker is present and cannot narrow: the submit would refuse,
        and the dry-run must say so.  This fails if the deferral is written
        over-broadly as `if not fetch: return None`.
        """

        avail = AVAIL + "\ngpu_b200x    mix host gpu:b200:8 gpu:b200:0 0/8/0/8"
        parts = PARTS + (
            "\nPartitionName=gpu_b200x AllowGroups=ALL AllowAccounts=ALL "
            "AllowQos=normal State=UP"
        )

        class TwoB200Client(FakeClient):
            def partition_access(self) -> str:
                return parts

        out, err = self._dry_run(
            warm=True, client=TwoB200Client(availability=avail), gpus="b200:1"
        )
        self.assertIn("--partition=gpu", out)
        self.assertIn("multiple partitions offer b200", err)
        self.assertNotIn("cannot resolve a partition", err)

    def test_a_denied_partition_is_flagged_rather_than_printed_silently(self) -> None:
        """The submit refuses a warm-cache DENY before dispatch; the dry-run
        used to print that partition with no warning at all -- a preview
        contradicting the thing it previews, in the state the docs call
        trustworthy.  h200x is offered only by priority_gpu, which this account
        cannot use, so it is still selected (eligibility is a tie-breaker, not a
        gate) and must now carry the warning the submit would raise."""

        avail = "priority_gpu mix host gpu:h200x:8 gpu:h200x:0 0/8/0/8"
        out, err = self._dry_run(
            warm=True, client=FakeClient(availability=avail), gpus="h200x:1"
        )
        self.assertIn("--partition=priority_gpu", out)  # still printed
        self.assertIn("not available to your account", err)  # ...but flagged


class DedicatedPartitionTests(unittest.TestCase):
    def test_scavenge_and_devel_are_not_dedicated(self) -> None:
        self.assertFalse(_is_dedicated_gpu_partition("scavenge"))
        self.assertFalse(_is_dedicated_gpu_partition("scavenge_gpu"))
        self.assertFalse(_is_dedicated_gpu_partition("gpu_devel"))
        self.assertTrue(_is_dedicated_gpu_partition("gpu_h200"))
        self.assertTrue(_is_dedicated_gpu_partition("gpu"))

    def test_custom_globs_replace_the_default(self) -> None:
        globs = ("preempt*", "*-short")
        self.assertFalse(_is_dedicated_gpu_partition("preempt_gpu", globs))
        self.assertFalse(_is_dedicated_gpu_partition("gpu-short", globs))
        # Names matched only by the built-in default are dedicated under a custom set.
        self.assertTrue(_is_dedicated_gpu_partition("scavenge_gpu", globs))
        self.assertTrue(_is_dedicated_gpu_partition("gpu_h200", globs))


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

    def test_type_only_on_preemptible_or_short_partitions_names_them(self) -> None:
        # l40s exists only on scavenge_gpu and gpu_devel here: no steady partition
        # offers it, so the tool refuses to auto-pick a preemptible/short pool but
        # NAMES them (with -p guidance), so the refusal agrees with `avail --for`
        # rather than claiming nothing offers the type.
        avail = AVAIL + "\nscavenge_gpu mix host gpu:l40s:4 gpu:l40s:0 0/4/0/4"
        avail += "\ngpu_devel    mix host gpu:l40s:4 gpu:l40s:0 0/4/0/4"
        client = FakeClient(availability=avail)
        with self.assertRaisesRegex(
            ConfigInvalid,
            r"no steady partition offers l40s GPUs.*preemptible.*gpu_devel, scavenge_gpu",
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
