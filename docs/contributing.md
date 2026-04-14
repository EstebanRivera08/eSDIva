# Contributing to PyField

Thank you for your interest in contributing to PyField!

## Development setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/EstebanRivera08/PyField.git
   cd PyField
   ```

2. **Install dependencies** (including dev tools):
   ```bash
   uv sync
   ```
3. **Install just tool** 
    ```bash
    uv tool install rust-just
    ```
4. **Run the test suite**:
   ```bash
   just test
   ```
5. **Run pre-commit hooks**:
   ```bash
   just pre-commit
   ```

## Code style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) (Black-compatible)
- **Linter**: Ruff (configured in `pyproject.toml`)
- **Type checker**: [ty](https://github.com/astral-sh/ty)
- **Docstrings**: [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html)

Pre-commit hooks enforce all of these automatically.

## Running tests

```bash
# Run all tests
just test

# Run with verbose output
just test-verbose

# Run only unit tests
uv run pytest tests/unit/ -v

# Run only integration tests
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest tests/ --cov=pyfield --cov-report=term-missing
```

## Adding a new transducer type

1. Create a new class inheriting from `TransducerBase` in the appropriate file
   under `src/pyfield/transducers/`
2. Implement `_compute_element_centers()` to define element positions
3. Implement `_build_subdivisions()` to generate rectangular patches
4. Export the new class in `src/pyfield/transducers/__init__.py`
5. Add tests in `tests/unit/test_transducers/`
6. Add documentation in `docs/api/transducers.md`

## Pull request workflow

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `just pre-commit` to verify code quality
4. Run `just test` to verify all tests pass
5. Open a PR with a clear description of the changes

