---
name: esdiva-contribute
description: Report a problem with eSDIva or propose a change to it — file a bug report or feature request with a minimal reproducer, or prepare a small, reviewable pull request that respects the package's physics, documentation and testing conventions. Use when the user says something in eSDIva is wrong, missing, confusing or slow, wants to open an issue, wants to contribute a fix or feature, or asks how to get a change accepted upstream.
---

# Contributing to eSDIva

Repository: <https://github.com/EstebanRivera08/eSDIva> · Issues:
<https://github.com/EstebanRivera08/eSDIva/issues>

eSDIva is a physics package close to release. A change that is merely plausible is
not good enough — a wrong result here becomes a wrong result in someone's paper. So
contributions are small, checked, and explain their acoustics. Two flows follow.

**Before either flow:** confirm the behaviour on the latest release. Reinstall
(`pip install -U esdiva`, or `uv sync` in a checkout) and re-run. A surprising number
of "bugs" are a version behind.

---

## Flow A — report a problem

The maintainers cannot fix what they cannot run. A minimal reproducer is worth more
than any description.

**1. Reduce it.** Cut the script until removing one more line makes the problem
disappear: one probe, the smallest grid or scatterer set that still shows it, a
seeded RNG, no plotting, no file I/O, under ~30 lines. Then **run the reduced
script** and paste its actual output. Never file a reproducer you have not executed.

**2. Separate observation from explanation.** Write what you saw and what you
expected. If you have a theory about the cause, mark it as a theory and say what
would confirm it. A confidently wrong cause sends the fix in the wrong direction —
"the PSF is wide because the pitch is 2λ" is a hypothesis until a control run with a
different pitch shows the width follow it.

**3. Collect the environment.**

```bash
python -c "import esdiva; print(esdiva.__version__)"
python -c "import sys, numpy, numba; print(sys.version); print(numpy.__version__, numba.__version__)"
```

**4. File it** against the repository's issue forms — a bug report asks for what
happened, the minimal reproducer, an **area** (`transducers`, `emission`,
`reception`, `hsir (SIR core)`, `beamforming`, `attenuation`, `plotting`, `io`,
`brain atlas`, `other`), the version and the OS/Python. Feature requests use their
own form.

```bash
gh issue create --repo EstebanRivera08/eSDIva --title "reception: <one-line symptom>" --body-file issue.md
```

Prefer the web form (`https://github.com/EstebanRivera08/eSDIva/issues/new/choose`)
if you are unsure which fields the template wants — it enforces them.

**Good title:** the area, then the symptom — `reception: pulse_echo_rf t0 shifts by
one sample when downsampling=2`. **Bad title:** "doesn't work".

**Not a bug report:** "how do I…" questions, and anything where the physics might
simply be right. Check `references/physics.md` in the `esdiva-simulate` skill first;
several classic surprises (a PSF that is too wide without impulse responses, an
image displaced by half a pulse length from re-applying the lag, non-Rayleigh
speckle from too few scatterers) are documented behaviour, not defects.

---

## Flow B — propose a change

**Small and understandable wins.** One concern per pull request, ideally under ~150
changed lines. A large refactor, however good, will sit unreviewed; the same work
split into three focused PRs lands. If a change grows while you write it, stop and
split it.

### Ask before you start, for any of these

Open an issue and agree the approach first — do not send an unsolicited PR:

- anything inside `src/esdiva/hsir/` (the SIR core, including the Numba kernels)
- a change to a public API's signature, defaults or return convention
- a new runtime dependency
- a change to a default method (`"auto"`, `"spectral"`) or to the `t0` convention
- a performance rewrite whose benefit you have not measured

Everything else — a documentation fix, a clearer docstring, a new test, a bug fix
with a failing test, a new plotting option, a new transducer geometry — can go
straight to a PR.

### Setup

```bash
git clone https://github.com/<you>/eSDIva.git && cd eSDIva
uv sync
uv tool install rust-just
just test          # confirm a clean baseline BEFORE you change anything
```

Work on a branch in your fork; never commit to `main`.

### While writing the change

- **Docstrings are for ultrasound researchers, not programmers.** Explain the
  physics first and inline: name the quantity, give its units, say why. Every
  docstring and comment must stand alone — **never cite a markdown file** (no "see
  `ARCHITECTURE.md`", no "see the module docstring"); those drift and the reader may
  not have them. Private functions get the same care.
- **Comment the why**, never the obvious. NumPy docstring format, `list[...]` /
  `tuple[...]` syntax, array shapes as `(X, Y, Z) numpy.ndarray`.
- **A physics claim needs a test, not a sentence.** If the change rests on "this is
  the correct delay sign" or "this makes the SIR converge", encode it as an
  assertion or a comparison against a reference (Field II, an analytic case, a naive
  implementation), so the claim survives the next refactor.
- **Match the surrounding style.** Absolute imports, `pathlib`, vectorised NumPy,
  `snake_case` / `PascalCase` / `_private`, specific exceptions with descriptive
  messages, `warnings.warn` for non-critical issues.
- **Tests must be able to fail.** A test that passes on garbage output is worse than
  no test. Prefer comparison against a reference implementation; use small arrays
  and a seeded RNG; test public API only. Use `numpy.testing.assert_allclose` with a
  real tolerance — float32 non-associativity and the SDI double cumsum mean exact
  equality is wrong (~0.004 % of peak is the expected SIR agreement).
- **Docs live in `docs/`** (a Zensical/MkDocs site). If behaviour changed, update the
  matching page and add new pages to `nav` in `zensical.toml`.
- **Numba caching bites.** After editing a kernel, clear the cache or your fix
  appears to do nothing:
  `Get-ChildItem -Path "src\esdiva\hsir\__pycache__" -Filter "*.nb?" | Remove-Item -Force`

### Before opening the PR

```bash
just pre-commit    # ruff check, ruff format, ty, codespell, numpydoc
just test          # full suite with coverage
```

Both must pass. On Windows `numpydoc` needs `PYTHONUTF8=1`; `just pre-commit` sets it.

Commit messages follow Commitizen: `<type>(<scope>): <summary>` with type in
`feat|fix|docs|style|refactor|perf|test|chore` and scope in `hsir|transducers|
emission|reception|attenuation|beamforming|plotting|utilities|io|atlas|docs|tests`.

```
fix(reception): keep t0 sample-aligned when downsampling

The decimation filter shifted the time origin by half a tap, so every
beamformed image sat one sample deep. Reference t0 to the filter's group
delay and assert the on-axis lag stays zero.
```

The pull-request template asks you to confirm: pre-commit passes, tests pass, tests
added or updated, docstrings physics-first and self-sufficient, `docs/` updated if
behaviour changed, and every physics claim backed by a check rather than asserted.
Fill it honestly — an unchecked box with an explanation is fine; a checked box that
is not true is not.

### Reviewing your own diff first

Read it as the maintainer would: is it one concern? Would a physicist understand the
new docstrings without opening anything else? Does a test fail if the change is
reverted? Is anything in it a claim rather than a check?
