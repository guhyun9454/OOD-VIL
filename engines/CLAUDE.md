<!-- Parent: ../CLAUDE.md -->

> **Instruction chain:** Read the parent [`../CLAUDE.md`](../CLAUDE.md) before this file; its broader instructions also apply to this subtree.

<!-- Generated: 2026-05-03 | Updated: 2026-05-19 -->

# OOD-VIL/engines

## Purpose
Training engines for the FT and ICON baselines. Each engine implements the full training loop — forward pass, loss computation, optimizer step, and OOD-score training — for its respective method.

## Key Files
| File | Description |
|------|-------------|
| `FT.py` | Fine-Tuning (FT) baseline engine: standard cross-entropy fine-tuning per task with post-hoc OOD scoring at test time. |
| `ICON.py` | ICON baseline engine: extends FT with in-context OOD training using DOS-synthesised outliers. |
| `custom_vit/` | Custom ViT modifications needed by ICON (attention hook insertions, etc.). |

## For AI Agents

### Working In This Directory
- `FT.py` is the simplest baseline; start here to understand the training loop before reading `ICON.py`.
- `ICON.py` calls `DOS/dos_sampler.py` during each training batch — DOS must be initialised before the engine starts.
- Entrypoint is `main.py` in the parent repo; it selects the engine class based on `--method` flag.

## Dependencies

### Internal
- `DOS/dos_sampler.py` — used by `ICON.py`.
- `OODdetectors/` — attached to the model post-training for inference.
- `continual_datasets/` — data fed to the engine.

<!-- MANUAL: -->
