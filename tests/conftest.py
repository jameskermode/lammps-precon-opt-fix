"""Pytest configuration for the lammps_precon stage tests.

LAMMPS + Symmetrix/Kokkos corrupt the heap during interpreter teardown (a
known ``LAMMPSlib`` issue — see the workaround in ``test_mace_lammps.py``).
The actual computations complete correctly; only process shutdown crashes.
We bypass the crashy teardown with a hard exit — done in ``pytest_unconfigure``
so it runs *after* pytest has printed the summary (including any failure
tracebacks) but *before* the real interpreter teardown.
"""
from __future__ import annotations

import os
import sys

import pytest

_RESULT = {"exitstatus": 1}


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    passed = session.testscollected - session.testsfailed
    print(f"\n[lammps_precon] {passed}/{session.testscollected} tests passed "
          f"— hard-exiting to skip the LAMMPS/Kokkos teardown crash")
    _RESULT["exitstatus"] = exitstatus


def pytest_unconfigure(config):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if _RESULT["exitstatus"] == 0 else 1)
