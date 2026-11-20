---
name: esdiva
description: The /esdiva front door. Classifies any eSDIva request — simulation, plotting, or physics — and routes it to the matching expert, so the user asks one place and lands with the right specialist. Use whenever someone types /esdiva, or mentions eSDIva, a spatial impulse response, an acoustic pressure field, a transducer aperture, a point spread function, RF channel data, delay-and-sum, moving/compounding a simulated field, a 3-D PyVista scene, or an ultrasound imaging simulation, without naming which part they need.
---

# eSDIva — front door

One job: read what the user actually wants and route it to the expert that does it
best. Do not ask the user which specialist to use — decide and go. A request with two
parts (explain the physics, then simulate it) routes in sequence.

## Know the API before touching source

Read `../esdiva-simulate/references/api-map.md` first (relative to this skill's
folder): every public signature for emission, reception, transducers, plotting,
beamforming and I/O, with a one-line purpose, its `file:line`, and a table of which
module owns what. Answer API questions from it directly; open source only at the
`file:line` it gives. When spawning an expert, tell it to read the map first too.

## Route

| The request is about… | Route to |
|---|---|
| Building a transducer, running an emission field or pulse-echo RF, beamforming, feeding RF to a custom/USTB beamformer, writing the simulation script | **`esdiva-sim` agent** |
| A figure appearing (desktop/notebook/headless), a 3-D PyVista scene, saving a still/movie, moving or compounding pressure volumes (`Field`), backend trouble | **`esdiva-plot` agent** |
| Why a result looks the way it does, fact-checking an SIR/SDI claim, method/sampling/`t0`, Field II equivalence, whether a study is in scope at all | **`esdiva-physics` agent** |

**On Claude Code**: spawn the matching agent with the Agent tool
(`subagent_type: "esdiva-sim" | "esdiva-plot" | "esdiva-physics"`), passing the user's
request. Each expert uses graphify to scope the package, reads its own references, and
cites `.claude/rules/*` and the gotcha list.

**On other hosts** (no custom agents): invoke the portable skill instead — `esdiva-sdi`
for physics/architecture and fact-checking, `esdiva-simulate` for hands-on simulation
and plotting, `esdiva-contribute` for issues and PRs. These travel with the package and
hold the same references the agents read.

## Common ground every route shares
- eSDIva is a **linear wave in a homogeneous fluid** with **Born point-scatterer**
  echoes. Skull, tissue layers, sound-speed maps, multiple scattering, harmonics, HIFU
  dose are outside the model (Field II's limits too) — name the reason and offer the
  nearest in-scope question rather than faking it.
- Physics first, code second. Units are millimetres in the public API.
- Don't assert a physical cause you haven't tested — label it a hypothesis and name the
  discriminating run.
