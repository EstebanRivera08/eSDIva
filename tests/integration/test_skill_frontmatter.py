"""The skills in `skills/` must stay loadable by every assistant that reads them.

Claude Code, OpenAI Codex and OpenCode all consume the same `SKILL.md` format, but
OpenCode applies the strictest validation: the front matter may contain only a known
set of keys, `name` must match the containing directory and the slug pattern
`^[a-z0-9]+(-[a-z0-9]+)*$`, and `description` is capped at 1024 characters. A skill
that violates any of those is silently skipped rather than reported, so the contract
is checked here instead of in someone's editor.
"""

import re
from pathlib import Path

import pytest

SKILLS = sorted((Path(__file__).parents[2] / "skills").glob("*/SKILL.md"))
ALLOWED_KEYS = {"name", "description", "license", "compatibility", "metadata"}
NAME_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _front_matter(path):
    """Parse the `key: value` block between the opening `---` fences.

    Deliberately a hand-rolled reader rather than a YAML parser: the block is a
    handful of flat scalars, and the check must not depend on a package that is
    only present in the environment by accident.
    """
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} must open with YAML front matter"
    block = text.split("---\n", 2)[1]
    return dict(
        (k.strip(), v.strip())
        for k, _, v in (
            line.partition(":") for line in block.splitlines() if line.strip()
        )
    )


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_skill_front_matter_is_portable(skill):
    meta = _front_matter(skill)
    assert set(meta) <= ALLOWED_KEYS, f"unsupported keys: {set(meta) - ALLOWED_KEYS}"
    assert meta["name"] == skill.parent.name, "name must match its directory"
    assert NAME_PATTERN.fullmatch(meta["name"]), "name must be a lowercase slug"
    assert 1 <= len(meta["description"]) <= 1024, "description capped at 1024 chars"


def test_skills_exist():
    assert SKILLS, "no skills found under skills/"
