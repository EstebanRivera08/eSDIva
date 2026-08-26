# SonDI Coding Guidelines

## Documentation & Comment Philosophy — Audience First (READ FIRST)

**SonDI is written for ultrasound researchers and students, not for programmers.**
The readers of this code are physicists and engineers who want to understand and check
the *implementation of the acoustics* — the SIR method, the pulse-echo chain, the
beamforming — so they can trust it, reproduce it, and extend it. Every docstring and
every comment is written for that reader. This is an explicit, non-negotiable design
principle of the package, not a style preference.

Rules that follow from this:

1. **Self-sufficient — never cite markdown files.** A docstring or comment must stand
   entirely on its own. **Do NOT reference markdown documents** (`ARCHITECTURE.md`,
   `PE_SDI_kernel_analysis.md`, `CLAUDE.md`, READMEs, "the module docstring", papers)
   from any docstring or comment — those files drift, get renamed, and the reader may
   not have them open. State the idea inline in one or two sentences so the reader
   understands *what* the method computes and *why* without opening anything else.
   A bare "see X" or "Full rationale: <file>.md" is forbidden; write the rationale here.
2. **Physics first, code second.** Explain the acoustic meaning before the array
   mechanics. Name the physical quantity (SIR, two-way delay, apodization, depth bin),
   give its units, and connect it to the equations in the module/`CLAUDE.md`. A reader
   should map the code to the textbook, not reverse-engineer the textbook from the code.
3. **Concise and clear.** Short, plain sentences. No filler, no restating the signature
   in prose. One or two sentences of intent beat a paragraph. If a method needs a long
   explanation, the *method* is probably doing too much.
4. **Easy to follow.** Define symbols the first time they appear in a docstring
   (`h_tx` = transmit spatial impulse response). Prefer the notation used in `CLAUDE.md`
   / the physics rules so the whole package reads consistently. Spell out a step the
   reader could not infer (a sign, a factor of `dt`, a `fs` scaling, a convention).
5. **Private methods get the same care.** A researcher auditing the code reads
   `_compute_rf_inner` and the kernels too. Private does not mean undocumented — it means
   "internal", and the *why* still matters. (Skip docstrings only on trivial helpers
   whose signature is genuinely self-evident, per "Readability vs Brevity" below.)
6. **Comments explain the physics/why, never the obvious.** Comment the non-obvious
   acoustic or numerical reason (why a `dt` factor, why `float64` accumulation, why this
   delay sign), not what the line literally does.

When in doubt, ask: *"Could an ultrasound PhD student who has never seen this file read
this docstring and understand the physics being computed?"* If not, rewrite it.

## Doubt-Driven Development (physics claims & delicate implementations)

A confident answer is not a correct one — especially deep into a long session, where
early hypotheses quietly calcify into "facts". A wrong-but-confident causal claim about
the acoustics (blaming an artefact on grating lobes, on pitch, on a missing impulse
response, ...) is worse than no answer: it steers a researcher into a physics
misunderstanding that survives in reports, docs, and memory long after the session.
(Precedent: example21's zeus10 image problems were confidently attributed to 2λ pitch
and to the impulse-response handling; a later 10 MHz rerun falsified both.)

Rules:

1. **Never assert a physical cause you have not tested.** Until a discriminating
   experiment (isolation run, control simulation, parameter sweep) has excluded the
   plausible alternatives, label the explanation explicitly as a *hypothesis*.
2. **Diagnoses written into docs, reports, or memory must record the discriminating
   test alongside the conclusion** — a claim without its falsification path is
   opinion, not a finding.
3. **For non-trivial questions** — physics interpretation, artefact diagnosis,
   delicate implementation choices — run the doubt cycle before asserting:

Doubt cycle:
- [ ] Step 1: CLAIM — wrote the claim + why-it-matters
- [ ] Step 2: EXTRACT — isolated artifact + contract, stripped reasoning
- [ ] Step 3: DOUBT — invoked fresh-context reviewer with adversarial prompt
- [ ] Step 4: RECONCILE — classified every finding against the artifact text
- [ ] Step 5: STOP — met stop condition (trivial findings, 3 cycles, or user override)

Step 3's reviewer must be *fresh-context* (a subagent that has not seen the reasoning
that produced the claim), prompted adversarially: "find why this claim is wrong or
untested". Step 4 classifies each finding as refutes / weakens / irrelevant against
the extracted artifact, not against the original reasoning.

## Project Status

Close to release. Be careful with core engine `hsir`. Transducers and utilities
under constant development — suggest breaking API changes when they improve design
and readability for adoption.

## Build/Lint/Test Commands

Uses [uv](https://docs.astral.sh/uv/) for dependencies, [just](https://github.com/casey/just) as command runner.

### Package Management
- `uv run <script.py>` - Run python scripts.
- `uv add <package>` - Add new dependencies.

### Build & Environment
- `uv sync` - Install dependencies and sync virtual environment
- `uv build` - Build package

### Documentation
- `just docs` (or `just d`) - Build documentation using Zensical
- `just clean-docs` (or `just cd`) - Clean documentation build directory and generated API files

### Linting, Formatting & Type Checking
- `just pre-commit` (or `just pc`) - Run all pre-commit hooks (recommended)
- `uv run ruff check . --fix` - Ruff linter with auto-fix
- `uv run ruff format .` - Format code with Ruff
- `uv run ty check src/` - ty type checking
- `uv run codespell` - Spell checker

Pre-commit hooks: ruff-check, ruff-format, ty, codespell, numpydoc-validation.

> **Windows note:** `numpydoc-validation` needs `PYTHONUTF8=1` on Windows (cp1252 vs
> UTF-8 math symbols in docstrings). `just pre-commit` sets this automatically.

### Testing
- `just test` (or `just t`) - Run all tests with coverage
- `just test-verbose` (or `just tv`) - Verbose test output
- `just generate-baselines` - Regenerate visual regression test baselines (pytest-mpl)
- `uv run pytest path/to/test_file.py` - Single test file
- `uv run pytest path/to/test_file.py::TestClass::test_method` - Single test

## Code Style

### Imports
- Absolute imports: `from sondi.io import AUTCDAT`
- Group: standard library, third-party, local
- Type-only: `from typing import TYPE_CHECKING`

### Formatting
- Ruff auto-formatting (Black compatible)
- Double quotes unless single needed for escaping

### Comments
1. Never duplicate code.
2. Good comments don't excuse unclear code.
3. Dispel confusion, not cause it.
4. Explain unidiomatic code.
5. Link to original source of copied code.
6. Link external references where helpful.
7. Add comments when fixing bugs.
8. `TODO:` prefix for incomplete implementations.
9. All comments end with period.
10. Explain complicated math briefly for understandability.

### Types
- `numpy.typing` / `npt.NDArray` for arrays
- `Literal` for string literals
- `TypedDict` for structured dicts
- `TypeAlias` for complex types

### Naming
- Functions/methods: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private: `_leading_underscore`

### Error Handling
- Specific exceptions: `ValueError`, `TypeError`, `FileNotFoundError`
- `warnings.warn()` for non-critical issues
- Validate inputs early with descriptive messages

### Documentation (NumPy Docstring Format)
- **Audience: see "Documentation & Comment Philosophy" at the top — every docstring is
  self-sufficient, physics-first, and written for ultrasound researchers/students.**
- Include Parameters, Returns, Raises sections
- Default values: `arg : type, default: value` or `arg : type, optional` for `None`
- Single backticks for inline code (Zensical/MkDocs style)
- Full package names: `xarray.DataArray` not `xr.DataArray`
- `list[...]`, `tuple[...]` syntax, not "list of..."
- Array shapes: `(X, Y, Z) numpy.ndarray`
- Multiple returns: separate lines per value, not `tuple[type1, type2]`
- Module constants: triple-quoted docstring immediately after constant
- Cross-references: `[name][sondi.module.path.name]` (mkdocs-style)
- No Sphinx-style references (`.. [1]`)

### Code Structure
- `pathlib.Path` for file operations
- Context managers for file handling
- List/dict comprehensions for simple transforms
- Single-responsibility functions
- NumPy vectorized operations over loops

### Readability vs Brevity (Open-Source Balance)
- Keep files short. Fewer lines = easier to audit and contribute to.
- Don't add docstrings to private helpers that have self-evident signatures.
- Don't expand comments that already say what the code does — only comment **why**.
- Prefer extracting a small helper over repeating 3+ lines, but don't abstract 1-time ops.
- Use consistent variable names across methods in same class (e.g. `pressure_flat` not mixed `Pressure_flat`).
- Inline comments: max 1 per logical block. No comment walls before code sections.
- Dispatch logic: one short comment per branch explaining **which mode** it handles.
- Dead code: delete immediately. No `# removed` markers, no `_unused` renames.
- Aim: a newcomer reads `__call__` and understands the full routing in under 2 minutes.

## Commit Message Convention (Commitizen)

Format: `<type>(<scope>): <short summary>`

Types: feat, fix, docs, style, refactor, perf, test, chore

Scopes: `hsir`, `transducers`, `emission`, `reception`, `attenuation`,
`beamforming`, `plotting`, `utilities`, `io`, `atlas`, `docs`, `tests`

## Testing Guidelines

### Philosophy
- **No useless tests**: Must fail if function returns garbage.
- **Concise**: No redundant tests. Each verifies something unique.
- **Public API only**: Don't test `_private` functions.

### What to Test
1. Edge cases: empty inputs, boundary conditions, special values.
2. Error validation: expected exceptions for invalid inputs.
3. Reference implementations: compare against known-correct (scipy, naive impl).

### Property-Based Tests
Only when no reference implementation exists (idempotence, commutativity, invariants).

### Structure
- pytest fixtures for reusable data (check `conftest.py` first)
- `numpy.testing.assert_allclose` for float comparisons
- `numpy.testing.assert_array_equal` for exact comparisons
- `pytest.raises` / `pytest.warns` for expected errors/warnings
- Small arrays, seeded RNG for reproducibility

### Visual Regression
- `@pytest.mark.mpl_image_compare` for plot tests
- `just generate-baselines` after intentional plot changes
- `uv run pytest --mpl` to enable image comparison

## Release

Use `/release NEW_VERSION` skill (`.claude/skills/release/SKILL.md`).
