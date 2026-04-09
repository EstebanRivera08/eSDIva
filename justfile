set windows-shell := ["pwsh.exe", "-c"]

# Print the help message.
@help:
    echo "Usage: just [RECIPE]\n"
    just --list

# Build documentation.
docs:
    uv run zensical build

# Serve documentation locally for development.
serve-docs:
    uv run zensical serve

# Clean documentation build artifacts.
clean-docs:
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .cache/, site/

# Run all tests (without visual regression comparison).
test:
    uv run pytest tests/

# Run all pre-commit hooks.
pre-commit:
    uv run prek run --all-files

# Run pre-commit and create and output .txt file
pre-commit-verbose:
    just pc > precommit_full.txt

# Run tests with coverage report.
coverage:
    uv run pytest tests/ --cov=pyfield --cov-report=term-missing --cov-report=html

# Aliases
alias d := docs
alias cd := clean-docs
alias sd := serve-docs
alias t := test
alias pc := pre-commit
alias cov := coverage
alias pcv := pre-commit-verbose
