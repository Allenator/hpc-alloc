from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.output import neutralize_stdout


class NeutralizeStdoutTests(unittest.TestCase):
    def test_redirects_stdout_and_closes_temporary_descriptor(self) -> None:
        stdout = SimpleNamespace(fileno=lambda: 1)
        with (
            patch("hpc_alloc.output.sys.stdout", stdout),
            patch("hpc_alloc.output.os.open", return_value=7) as open_devnull,
            patch("hpc_alloc.output.os.dup2") as duplicate,
            patch("hpc_alloc.output.os.close") as close,
        ):
            neutralize_stdout()

        open_devnull.assert_called_once_with("/dev/null", 1)
        duplicate.assert_called_once_with(7, 1)
        close.assert_called_once_with(7)

    def test_does_not_close_stdout_when_open_returns_its_descriptor(self) -> None:
        stdout = SimpleNamespace(fileno=lambda: 1)
        with (
            patch("hpc_alloc.output.sys.stdout", stdout),
            patch("hpc_alloc.output.os.open", return_value=1),
            patch("hpc_alloc.output.os.dup2") as duplicate,
            patch("hpc_alloc.output.os.close") as close,
        ):
            neutralize_stdout()

        duplicate.assert_not_called()
        close.assert_not_called()

    def test_stdout_without_usable_descriptor_is_ignored(self) -> None:
        unusable_stdout = (
            object(),
            io.StringIO(),
            SimpleNamespace(fileno=lambda: 9),
            SimpleNamespace(fileno=lambda: -1),
            SimpleNamespace(fileno=Mock(side_effect=TypeError("not a descriptor"))),
            SimpleNamespace(fileno=Mock(side_effect=ValueError("closed"))),
        )
        for stdout in unusable_stdout:
            with self.subTest(stdout=type(stdout).__name__):
                with (
                    patch("hpc_alloc.output.sys.stdout", stdout),
                    patch("hpc_alloc.output.os.open") as open_devnull,
                ):
                    neutralize_stdout()

                open_devnull.assert_not_called()

    def test_open_duplication_and_close_failures_are_best_effort(self) -> None:
        stdout = SimpleNamespace(fileno=lambda: 1)
        with (
            patch("hpc_alloc.output.sys.stdout", stdout),
            patch("hpc_alloc.output.os.open", side_effect=OSError("unavailable")),
        ):
            neutralize_stdout()

        with (
            patch("hpc_alloc.output.sys.stdout", stdout),
            patch("hpc_alloc.output.os.open", return_value=7),
            patch("hpc_alloc.output.os.dup2", side_effect=OSError("closed")),
            patch("hpc_alloc.output.os.close", side_effect=OSError("closed")) as close,
        ):
            neutralize_stdout()

        close.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()
