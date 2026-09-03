"""The agent-skill templates must keep running against the real package.

`skills/esdiva-simulate/templates/` ships the starting scripts an AI
assistant hands to a user: a CW beam profile, a transient wavefront, a pulse-echo
PSF, and a diverging-wave sequence beamformed both with `das_volume` and by hand.
They are documentation that executes, so an API change must break them here rather
than in a user's first session. Each is run as a subprocess with a non-interactive
Matplotlib backend; the only assertion is that it completes.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TEMPLATES = (
    Path(__file__).parents[2] / "skills/esdiva-simulate/templates"
)


@pytest.mark.parametrize("script", sorted(p.name for p in TEMPLATES.glob("*.py")))
def test_template_runs(script):
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(TEMPLATES / script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
