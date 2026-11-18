<!-- Parent: ../CLAUDE.md -->

> **Instruction chain:** Read the parent [`../CLAUDE.md`](../CLAUDE.md) before this file; its broader instructions also apply to this subtree.

<!-- Generated: 2026-05-03 | Updated: 2026-05-19 -->

# OOD-VIL/continual_datasets

## Purpose
Dataset wrappers for the FT/ICON baselines, implementing the OOD-VIL incremental-learning protocol. Structurally identical to `RanPAC/continual_datasets/` — replicated for repo independence.

## Key Files
| File | Description |
|------|-------------|
| `base_datasets.py` | Raw dataset loaders; reads from `/local_datasets/` on compute nodes. |
| `build_incremental_scenario.py` | Splits into tasks, assigns OOD labels for held-out classes. |
| `dataset_utils.py` | Shared transforms and label-remapping helpers. |

## For AI Agents

### Working In This Directory
- Dataset root must be `/local_datasets/` on Seraph compute nodes.
- Sync-fixes should be applied to both this file and `RanPAC/continual_datasets/` to keep baselines consistent.

## Dependencies

### External
- `/local_datasets/` — synced via `experiments/scripts/sync_dataset.sh`.

<!-- MANUAL: -->
