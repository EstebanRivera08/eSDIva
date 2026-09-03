---
icon: lucide/bot
---

# Using eSDIva with an AI assistant

The repository ships **agent skills**: Markdown files that teach a coding assistant
how eSDIva works — its conventions, its API, the physics behind the numbers, and the
mistakes that quietly produce wrong results.

They are optional. The package works without them.

## What they cover

**`esdiva-simulate`** — building a transducer (array, bowl, custom assembly, Field II
import), emission simulations (CW beam maps and transient wavefronts), pulse-echo RF
(PSF, phantoms, plane- and diverging-wave sequences, FMC), beamforming with the
built-in reconstructors *or* with your own, choosing a Matplotlib/PyVista backend
that actually displays something, and the SIR/SDI physics behind any of it. It
carries four runnable templates:

| Template | Answers |
|---|---|
| `emission_cw.py` | beam width, depth of field, sidelobe level |
| `emission_transient.py` | wavefront propagation, pulse shape, time of flight |
| `reception_psf.py` | the point spread function, and a check of the `t0` convention |
| `reception_sequence_das.py` | a diverging-wave acquisition, beamformed both with `das_volume` and by hand |

Templates are `# %%`-celled Python: run them as scripts, or open them as notebooks.
They are executed by the test suite on every CI run, so they cannot drift away from
the package.

**`esdiva-contribute`** — writing a bug report with a reduced, executed reproducer,
and preparing a small pull request that follows the project's physics, documentation
and testing conventions.

## Install

The skills live in `skills/` in the repository as plain `SKILL.md` files with
`name`/`description` front matter — the format Claude Code, OpenAI Codex and
OpenCode all read, so the same folder serves all three.

=== "Claude Code"

    ```
    /plugin marketplace add EstebanRivera08/eSDIva
    /plugin install esdiva
    ```

    Or, without the marketplace, copy the two skill folders into `~/.claude/skills/`
    (available in every project) or `.claude/skills/` (one project).

    **Check it loaded:** run `/plugin` and look for `esdiva` under installed
    plugins, or ask *"which eSDIva skills do you have?"*.

=== "OpenAI Codex"

    Codex scans `.agents/skills/` upward from the working directory, and
    `$HOME/.agents/skills/` for user-scoped skills.

    ```bash
    git clone --depth 1 https://github.com/EstebanRivera08/eSDIva /tmp/esdiva
    mkdir -p ~/.agents/skills && cp -r /tmp/esdiva/skills/esdiva-* ~/.agents/skills/
    ```

    ```powershell
    git clone --depth 1 https://github.com/EstebanRivera08/eSDIva $env:TEMP\esdiva
    New-Item -ItemType Directory -Force $HOME\.agents\skills
    Copy-Item -Recurse $env:TEMP\esdiva\skills\esdiva-* $HOME\.agents\skills\
    ```

    **Check it loaded:** run `/skills` in Codex, or type `$` to mention one —
    `esdiva-simulate` and `esdiva-contribute` should be listed.

=== "OpenCode"

    OpenCode reads `.opencode/skills/`, `~/.config/opencode/skills/`, and — being
    compatible with both other formats — `.claude/skills/` and `.agents/skills/`,
    project and global. The Codex command above therefore installs them for OpenCode
    as well; for a dedicated location:

    ```bash
    mkdir -p ~/.config/opencode/skills && cp -r /tmp/esdiva/skills/esdiva-* ~/.config/opencode/skills/
    ```

    **Check it loaded:** the skills appear in the session's skill list; asking a
    simulation question should pull `esdiva-simulate` in.

=== "Anything else"

    Copy `skills/esdiva-simulate/` into whatever your tool reads — `AGENTS.md`,
    `.cursor/rules/`, a Copilot instructions file, a prompt library. The routing
    table at the top of each `SKILL.md` points to the reference file for each kind
    of question, and every path inside a skill is relative to its own folder, so the
    structure survives the move.

## Using them

Skills are model-invoked: you do not call them by name. Ask the physical question
and the assistant pulls in what it needs.

> *"I have a 64-element linear array at 5 MHz, 0.3 mm pitch. Show me the beam
> profile at 30 mm and tell me the −6 dB width."*

> *"Simulate the PSF of my probe at three depths, then explain why it widens."*

> *"Give me RF channel data for a diverging-wave sequence — I want to beamform it
> myself with my own Fourier-domain code. What exactly is in the array?"*

If the assistant starts writing eSDIva code without mentioning units, `no_sub_x`, or
what `coords["t0"]` means, the skill is probably not loaded.

## What the skills instruct

- Start from a ready-made probe (`Domino()`, `Zeus_Matrix()`) unless you give
  element dimensions.
- Run a coarse 2-D plane before anything expensive, and say what a long run costs
  before launching it.
- Describe the output physically: pressure in pascals on an `(Nt, Nx, Ny, Nz)` grid,
  a `t0` that is already the beamforming reference.
- Treat an explanation for an artefact as a hypothesis until a control simulation
  rules out the alternatives. If you are told *why* something looks the way it does,
  ask what test would show that explanation is wrong.
