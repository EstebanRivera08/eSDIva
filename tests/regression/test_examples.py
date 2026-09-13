"""Every numbered example must still run end to end (opt-in: ``just test-examples``).

Examples are documentation that executes; an API change must break them here, not in a
user's hands. Each runs through `headless.py` (every plotter off-screen and closed after
one render, no figure saving). The multi-step studies in ``example21_*/`` and
``example22_*/`` are excluded — they acquire RF for hours by design. Numbers are pinned
separately by `test_golden.py`.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"
HEADLESS = Path(__file__).with_name("headless.py")
ENV = {**os.environ, "ESDIVA_SAVE_FIG": "0"}


@pytest.mark.examples
@pytest.mark.parametrize(
    "script", sorted(EXAMPLES.glob("example*.py")), ids=lambda p: p.stem
)
def test_example_runs(script):
    result = subprocess.run(
        [sys.executable, str(HEADLESS), str(script)],
        cwd=EXAMPLES,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-3000:]
