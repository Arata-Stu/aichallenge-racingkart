#!/usr/bin/env python3
"""Supervise one process group with Linux parent-death protection."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time


PR_SET_PDEATHSIG = 1


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: process_guard.py PARENT_PID COMMAND [ARG ...]")
    expected_parent = int(sys.argv[1])
    command = sys.argv[2:]
    child: subprocess.Popen[bytes] | None = None
    terminate_requested = False
    terminate_started_at: float | None = None

    def terminate_group(_signum: int, _frame: object) -> None:
        nonlocal terminate_requested, terminate_started_at
        terminate_requested = True
        terminate_started_at = terminate_started_at or time.monotonic()
        # The guard is the process-group leader. Ignore our own forwarded signal,
        # while the command and all of its descendants retain the default action.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            try:
                os.killpg(os.getpgrp(), signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, terminate_group)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    # Close the race where the parent died immediately before prctl().
    if os.getppid() != expected_parent:
        raise SystemExit("dashboard parent exited before the job guard was ready")
    if terminate_requested:
        raise SystemExit(143)

    child = subprocess.Popen(command)
    if terminate_requested:
        terminate_group(signal.SIGTERM, None)

    while child.poll() is None:
        if (
            terminate_requested
            and terminate_started_at is not None
            and time.monotonic() - terminate_started_at >= 5.0
        ):
            # This also terminates the guard. It is intentional: nothing in the
            # dedicated job process group may survive the grace period.
            os.killpg(os.getpgrp(), signal.SIGKILL)
        time.sleep(0.05)
    return_code = child.wait()
    raise SystemExit(return_code if return_code >= 0 else 128 - return_code)


if __name__ == "__main__":
    main()
