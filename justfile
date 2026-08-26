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

# Clean graphify non-crucial cache/state (keeps graph.json/html, GRAPH_REPORT.md, manifest.json).
clean-graphify:
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue graphify-out/cache/, graphify-out/cost.json, graphify-out/.graphify_labels.json, graphify-out/.graphify_root, graphify-out/.graphify_python

# Run all tests (without visual regression comparison).
test:
    uv run pytest tests/

# Run all pre-commit hooks.
pre-commit:
    $env:PYTHONUTF8 = "1"; uv run prek run --all-files

# Run pre-commit and create and output .txt file
pre-commit-verbose:
    just pc > precommit_full.txt

# Run tests with coverage report.
coverage:
    uv run pytest tests/ --cov=esdiva --cov-report=term-missing --cov-report=html

# Aliases
alias d := docs
alias cd := clean-docs
alias cg := clean-graphify
alias sd := serve-docs
alias t := test
alias pc := pre-commit
alias cov := coverage
alias pcv := pre-commit-verbose
