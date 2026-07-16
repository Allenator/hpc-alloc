from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.commands import cmd_avail


# `sinfo -h -N -O Partition,StateCompact,NodeHost,Gres,GresUsed,CPUsState` rows.
AVAIL = "\n".join(
    (
        "gpu_h200     idle host gpu:h200:8 gpu:h200:0 0/64/0/64",
        "priority_gpu idle host gpu:h200:8 gpu:h200:0 0/64/0/64",
        "day          idle host (null) (null) 0/32/0/32",
        "mystery      idle host (null) (null) 0/16/0/16",
    )
)
USER = "GROUPS ab1234 pi_lab01\nASSOC\npi_lab01||interactive,normal\n"
# priority_gpu demands the `priority` account ab1234 lacks; mystery has no rule.
PARTS = "\n".join(
    (
        "PartitionName=gpu_h200 AllowGroups=ALL AllowAccounts=ALL AllowQos=normal State=UP",
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


class FakeClient:
    def availability(self) -> str:
        return AVAIL

    def user_access(self, netid: str) -> str:
        return USER

    def partition_access(self) -> str:
        return PARTS


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            resolve_cluster=lambda _requested: "bouchet",
            identity=SimpleNamespace(netid="ab1234"),
        ),
        state=FakeState(),
    )


def _args(json_output: bool) -> SimpleNamespace:
    return SimpleNamespace(cluster=None, partition=None, probe=False, json=json_output)


class AvailEligibilityTests(unittest.TestCase):
    def _run(self, json_output: bool) -> str:
        transport = Mock()
        buffer = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, FakeClient()),
            ),
            redirect_stdout(buffer),
        ):
            code = cmd_avail(
                _args(json_output),
                ctx=_ctx(),
                paths=SimpleNamespace(),
                entrypoint=SimpleNamespace(),
            )
        self.assertEqual(code, 0)
        transport.bootstrap.assert_called_once_with("bouchet")
        return buffer.getvalue()

    def test_json_payload_carries_eligibility(self) -> None:
        payload = json.loads(self._run(json_output=True))["partitions"]
        self.assertIs(payload["gpu_h200"]["eligible"], True)
        self.assertIs(payload["priority_gpu"]["eligible"], False)
        self.assertIs(payload["day"]["eligible"], True)
        # A partition with no access rule falls open to unknown (null).
        self.assertIsNone(payload["mystery"]["eligible"])

    def test_text_table_has_eligible_column(self) -> None:
        output = self._run(json_output=False)
        lines = output.splitlines()
        header = lines[0]
        self.assertIn("ELIGIBLE", header)
        rows = {line.split()[0]: line for line in lines[1:] if line.split()}
        self.assertEqual(rows["gpu_h200"].split()[1], "yes")
        self.assertEqual(rows["priority_gpu"].split()[1], "no")
        self.assertEqual(rows["day"].split()[1], "yes")
        self.assertEqual(rows["mystery"].split()[1], "?")


if __name__ == "__main__":
    unittest.main()
