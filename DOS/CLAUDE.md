<!-- Parent: ../CLAUDE.md -->

> **Instruction chain:** Read the parent [`../CLAUDE.md`](../CLAUDE.md) before this file; its broader instructions also apply to this subtree.

<!-- Generated: 2026-05-03 | Updated: 2026-05-19 -->

# OOD-VIL/DOS

## Purpose
DOS (Diversified Outlier Synthesis) module — a data augmentation technique that generates synthetic OOD samples near the decision boundary to improve OOD detector training. Ported from the upstream DOS paper for integration into the FT/ICON pipeline.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package init. |
| `dos_sampler.py` | Core DOS sampler: generates synthetic outlier features by interpolating between in-distribution class means. |
| `dos.md` | Design notes and porting decisions for the DOS integration. |
| `DOS_PORTING_GUIDE.md` | Step-by-step guide for integrating DOS into a new baseline repo. |

## For AI Agents

### Working In This Directory
- DOS is used only in the `OOD-VIL/` baseline (FT + ICON) — not in RanPAC/ETE.
- `dos.md` and `DOS_PORTING_GUIDE.md` are design documents — read them before modifying `dos_sampler.py`.
- The `engines/ICON.py` and `engines/FT.py` call the DOS sampler during training.

## Dependencies

### Internal
- `engines/ICON.py`, `engines/FT.py` — consume the DOS sampler.

### External
- `torch`, `numpy`.

<!-- MANUAL: -->
