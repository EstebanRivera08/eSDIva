# AGENTS.md

Guidance for coding agents working in this repository (Codex, OpenCode, Cursor,
Claude Code, …). Two audiences, two sets of files.

## If you are helping someone *use* eSDIva

Read `skills/esdiva-simulate/SKILL.md`. It routes to focused references —
transducer geometry, emission, pulse-echo reception and custom beamformers,
plotting backends, and the underlying SIR physics — plus four runnable templates in
`skills/esdiva-simulate/templates/`. Those skills are also installable outside this
repository; see `skills/README.md`.

For reporting a bug or preparing a pull request, read
`skills/esdiva-contribute/SKILL.md`.

## If you are *modifying* this repository

Read `CLAUDE.md` (architecture map, commands, physics conventions, gotchas) and the
rule files it points to in `.claude/rules/`. They are plain Markdown and apply
whatever assistant you are.

The non-negotiables, in short:

- **Docstrings and comments are written for ultrasound researchers, not for
  programmers.** Explain the physics first, inline, with units. Every docstring must
  stand alone — never cite a Markdown file from code.
- **Never assert a physical cause you have not tested.** A diagnosis without its
  discriminating experiment is a hypothesis; label it as one.
- **Units:** public API in millimetres (`_mm` suffix), internals in SI.
- `just pre-commit` and `just test` must both pass before a change is proposed.
- Commit messages follow Commitizen: `<type>(<scope>): <summary>`.

## Commands

```bash
uv sync            # install and sync the environment
just test          # test suite with coverage
just pre-commit    # ruff, ty, codespell, numpydoc
just serve-docs    # build and serve the documentation locally
```
