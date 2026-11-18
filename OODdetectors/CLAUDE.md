<!-- Parent: ../CLAUDE.md -->

> **Instruction chain:** Read the parent [`../CLAUDE.md`](../CLAUDE.md) before this file; its broader instructions also apply to this subtree.

<!-- Generated: 2026-05-03 | Updated: 2026-05-19 -->

# OOD-VIL/OODdetectors

## Purpose
Post-hoc OOD score postprocessors for the FT / ICON baselines. Identical interface to `RanPAC/OODdetectors/` — the same 9 postprocessors are replicated here so each repo is self-contained.

## Key Files
| File | Description |
|------|-------------|
| `base_postprocessor.py` | Abstract base — `setup()` and `postprocess()` interface. |
| `ebo_postprocessor.py` | Energy-Based OOD (ENERGY score). |
| `gen_postprocessor.py` | GEN score (Generalised ENtropy). |
| `maxlogit_postprocessor.py` | MaxLogit score. |
| `ood_adapter.py` | Dispatcher keyed on `--ood_method` string. |
| `pro_gen_postprocessor.py` | PRO_GEN — prototype-guided GEN. |
| `pro2_ent_postprocessor.py` | PRO2 + entropy. |
| `pro2_msp_postprocessor.py` | PRO2 + MSP. |
| `pro2_tempscale_postprocessor.py` | PRO2 + temperature scaling. |

## For AI Agents

### Working In This Directory
- Always use `--ood_method MSP,MAXLOGIT,ENERGY,GEN,PRO_GEN` to run all five paper-reported detectors in one pass.
- This directory is structurally identical to `RanPAC/OODdetectors/` — sync bug-fixes across both if patching shared logic.

## Dependencies

### Internal
- `ood_adapter.py` — the single import point used by `main.py`.

<!-- MANUAL: -->
