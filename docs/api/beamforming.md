---
icon: lucide/git-merge
---

# Beamforming

Delay-and-sum RF post-processing. All DAS beamformers auto-apply
`coords["pulse_center_lag_s"]` and recover each event's transmit time reference —
see the [Reception user guide](../user-guide/reception.md#beamforming-note).

## das_volume

General Numba 3-D DAS (TX=RX). Each event carries `delays`/`apodization` plus a
`virtual_source_mm` (DW z<0 / focused z>0 / synthetic z≈0) or `angles_deg` (PW).

::: pyfield.beamforming.das_volume

## das_rca_volume

3-D DAS specialised for row-column (RCA) plane-wave sequences.

::: pyfield.beamforming.das_rca_volume

## DAS_focused_scanline

Single focused B-mode scan line.

::: pyfield.beamforming.DAS_focused_scanline

## envelope_db

Envelope detection and log compression to dB.

::: pyfield.beamforming.envelope_db
