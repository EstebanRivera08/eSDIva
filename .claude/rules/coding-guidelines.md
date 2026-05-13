# PyField Coding Guidelines

## Project Status

Close to release. Be careful with core engine `h_sir`. Transducers and utilities
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
- Absolute imports: `from pyfield.io import AUTCDAT`
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
- Include Parameters, Returns, Raises sections
- Default values: `arg : type, default: value` or `arg : type, optional` for `None`
- Single backticks for inline code (Zensical/MkDocs style)
- Full package names: `xarray.DataArray` not `xr.DataArray`
- `list[...]`, `tuple[...]` syntax, not "list of..."
- Array shapes: `(X, Y, Z) numpy.ndarray`
- Multiple returns: separate lines per value, not `tuple[type1, type2]`
- Module constants: triple-quoted docstring immediately after constant
- Cross-references: `[name][pyfield.module.path.name]` (mkdocs-style)
- No Sphinx-style references (`.. [1]`)

### Code Structure
- `pathlib.Path` for file operations
- Context managers for file handling
- List/dict comprehensions for simple transforms
- Single-responsibility functions
- NumPy vectorized operations over loops

## Commit Message Convention (Commitizen)

Format: `<type>(<scope>): <short summary>`

Types: feat, fix, docs, style, refactor, perf, test, chore

Scopes: `h_sir`, `transducers`, `psimulation`, `plotting`, `utilities`,
`io`, `atlas`, `docs`, `tests`

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
