"""The agent API map must match the live public API, or agents read stale signatures."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_api_map_is_current():
    gen = runpy.run_path(str(ROOT / "scripts" / "gen_api_map.py"))
    on_disk = gen["OUT"].read_text(encoding="utf-8")
    assert on_disk == gen["render"](), "API map is stale: run `just api-map`."
