---
name: release
description: Create a new eSDIva release (version bump, commit, tag, push, build, release notes)
argument-hint: <new-version>
disable-model-invocation: true
---

Perform a full eSDIva release. The new version string is: `$ARGUMENTS`

Follow these steps in order. **Do not push, or take any other irreversible action,
until explicitly confirmed by the user.** The PyPI upload (Step 10) is done by the
user, never by you.

---

## Step 1 — Validate input

If no version was provided, ask the user for the version string (e.g. `0.2.0`).

Read the current version from `pyproject.toml`. Find the most recent tag:

```bash
git describe --tags --abbrev=0
```

This fails when no tag exists yet — that is fine, treat the previous tag as absent
and use the root commit as the range start in Step 6.

Recommend a version that matches the change set, and say why: for `0.x`, any
break in a public contract (a changed return convention, a dependency moved into
an extra, a renamed argument) is a **minor** bump, not a patch. State the breaking
changes explicitly so the user can overrule the number.

---

## Step 2 — Update the version

### `pyproject.toml`
Replace `version = "OLD"` with `version = "NEW"`.

That is the only version reference in the repo. The citation in `README.md` and
`docs/citing.md` points at the **paper**, not the software release — do not add a
version or bump a year there.

---

## Step 3 — Sync lock file

```bash
uv sync
```

---

## Step 4 — Run pre-commit checks

```bash
just pre-commit
```

Fix any failures before continuing.

---

## Step 5 — Create version bump commit

Stage only `pyproject.toml` and `uv.lock`.

Commit message:

```
chore: bump version to vNEW
```

---

## Step 6 — Create annotated tag

Collect the commit list since the previous tag, excluding the version bump commit
itself:

```bash
git log vPREV..HEAD~1 --oneline     # when a previous tag exists
git log HEAD~1 --oneline            # first release: everything up to the bump
```

Group by prefix into sections (omit empty sections):

| Commit prefix        | Section heading     |
|----------------------|---------------------|
| `feat`               | **New features**    |
| `fix`                | **Bug fixes**       |
| `docs`               | **Documentation**   |
| `refactor`, `perf`   | **Improvements**    |
| `test`, `chore`, `style` | **Other**       |

Put any breaking change first, under **Breaking changes**, with the migration a
user has to perform — not just what changed.

Use the grouped list as SUMMARY in the tag message below:

```
eSDIva vNEW

SUMMARY (bullet list, one line per commit, strip the conventional commit prefix)
```

Create the tag:

```bash
git tag -a vNEW -m "$(cat <<'EOF'
<the message above>
EOF
)"
```

---

## Step 7 — Build the distribution

```bash
just build
```

Check that `dist/` holds exactly the new `.whl` and `.tar.gz` — a stale artifact
from an older version left in `dist/` will be uploaded by `twine upload dist/*`.

---

## Step 8 — Review and confirm

Show the user:

1. The full commit diff: `git show HEAD`
2. The tag message: `git tag -n99 vNEW`
3. The built artifacts: `ls dist/`

Then ask: **"Ready to push commit and tag to origin? (yes / no)"**

Do **not** push until the user explicitly says yes.

---

## Step 9 — Push (user only)

Hand the user the commands; do not run them.

```bash
git push origin main
git push origin vNEW
```

---

## Step 10 — PyPI upload (user only)

Hand the user the command; do not run it. PyPI refuses a re-upload of a version
that already exists, so a mistake here costs a version number.

```bash
just publish        # twine upload dist/*
```

Remind them that `pip install esdiva` then resolves to the highest version, so
0.2.0 supersedes 0.1.0 for new installs while `esdiva==0.1.0` stays available.

---

## Step 11 — GitHub release message

Generate and display the following for the user to paste into the GitHub release
UI. Use the same grouped commit list from Step 6 (omit **Other** unless notable).

```markdown
## What's new in vNEW

### Breaking changes
- ...

### New features
- ...

### Bug fixes
- ...

### Documentation
- ...

### Improvements
- ...

```
