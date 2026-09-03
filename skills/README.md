# eSDIva agent skills

Portable [Agent Skills](https://code.claude.com/docs/en/skills) for
[eSDIva](https://github.com/EstebanRivera08/eSDIva), the Tupholme–Stepanishen
ultrasound field simulator. Plain Markdown with `name` / `description` front matter —
the format Claude Code, OpenAI Codex and OpenCode all read.

- **`esdiva-simulate`** — build transducers, run emission (CW / transient) and
  pulse-echo RF simulations, beamform the RF or feed it to your own beamformer, get
  a figure to actually appear, and understand the SIR/SDI physics behind the result.
  Four runnable templates included.
- **`esdiva-contribute`** — file a useful bug report (reduced, executed reproducer)
  or prepare a small, reviewable pull request.

## Install

### Claude Code

```
/plugin marketplace add EstebanRivera08/eSDIva
/plugin install esdiva
```

Or without the marketplace, copy a skill folder into `~/.claude/skills/` (all
projects) or `.claude/skills/` (one project). Check with `/plugin`.

To test a local checkout before publishing:

```bash
claude plugin validate .            # manifests, versions, source paths
claude --plugin-dir .               # load this checkout as a plugin for one session
claude plugin marketplace add ./    # or register the local dir as a marketplace
```

### OpenAI Codex

Codex scans `.agents/skills/` upward from the working directory, and
`$HOME/.agents/skills/` for user-scoped skills.

```bash
git clone --depth 1 https://github.com/EstebanRivera08/eSDIva /tmp/esdiva
mkdir -p ~/.agents/skills && cp -r /tmp/esdiva/skills/esdiva-* ~/.agents/skills/
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force $HOME\.agents\skills
Copy-Item -Recurse .\skills\esdiva-* $HOME\.agents\skills\
```

Swap `~/.agents/skills` for `.agents/skills` inside a project to share the skills
with a team through that repository. Confirm with `/skills` in Codex, or type `$`.

### OpenCode

OpenCode searches `.opencode/skills/`, `~/.config/opencode/skills/`, and — being
compatible with both other formats — `.claude/skills/`, `~/.claude/skills/`,
`.agents/skills/` and `~/.agents/skills/`. So the Codex command above installs them
for OpenCode too; otherwise:

```bash
mkdir -p ~/.config/opencode/skills && cp -r /tmp/esdiva/skills/esdiva-* ~/.config/opencode/skills/
```

### Anything else (Cursor, Copilot, Gemini CLI, a prompt library)

The skills are ordinary Markdown. Copy `esdiva-simulate/` into whatever your tool
reads — `AGENTS.md`, `.cursor/rules/`, an instructions file — or paste `SKILL.md`
and let the assistant open the reference it names. The routing table at the top of
each `SKILL.md` says which file answers which question, so the structure survives
the move.

## Layout

```
esdiva-simulate/
  SKILL.md            routing table, onboarding flow, the rules that decide correctness
  references/         transducers · emission · reception · visualization · physics
  templates/          four runnable scripts
esdiva-contribute/
  SKILL.md            issue workflow + small-pull-request workflow
```

Paths inside a skill are relative to its own `SKILL.md`, so the folder can be moved
anywhere as a unit. Nothing in either skill is specific to one assistant.

The templates are ordinary Python with `# %%` cell markers: run them as scripts, or
open them as notebooks (VS Code, `jupytext`).

## Keeping them honest

`tests/integration/test_skill_templates.py` in the main repository executes every
template on each CI run, so an API change breaks the build instead of breaking a
user's first session.
