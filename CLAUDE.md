<!-- Parent: ../CLAUDE.md -->

> **Instruction chain:** Read the parent [`../CLAUDE.md`](../CLAUDE.md) before this file; its broader instructions also apply to this subtree.

<!-- Generated: 2026-04-23 | Updated: 2026-05-19 -->

# OOD-VIL (baseline repo)

## Purpose
Home of the **FT** and **ICON** baselines on the `main` branch; also hosts **RainbowPrompt** on the `integrate-RainbowPrompt` branch (integrated from the removed RainbowPrompt upstream clone — see parent [`../CLAUDE.md`](../CLAUDE.md)). Post-hoc OOD detectors run on top.

## Key Files
| File | Description |
|------|-------------|
| `main.py` | Entrypoint for FT / ICON / RainbowPrompt VIL runs. |
| `models.py` | Model definitions (ICON architecture, prompt-based variants). |
| `utils.py` | Shared helpers. |
| `requirements.txt` | Python dependencies. |
| `OE.md` | Notes on Outlier Exposure experiments (pivot history). |
| `README.md`, `LICENSE` | Upstream README + license. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `engines/` | Per-method training engines — FT, ICON, and (on the right branch) RainbowPrompt. |
| `OODdetectors/` | MSP / MAXLOGIT / ENERGY / GEN / PRO_GEN post-hoc detectors. |
| `continual_datasets/` | iDigits / DomainNet / CLEAR loaders + OOD dataset shims. |
| `experiments/` | Run configs + scripts. |
| `DOS/` | Density-of-States (DOS) related scripts (exploratory). |

## For AI Agents

### Working In This Directory
- **Branch awareness**: `main` has FT/ICON; RainbowPrompt lives on `integrate-RainbowPrompt`. Check `git branch` before running.
- **FT / ICON run template** (from `experiments/baseline_commands.md`):
  ```bash
  python main.py --dataset <iDigits|DomainNet> --IL_mode vil --num_tasks <20|30> \
      --ood_dataset <EMNIST|CORe50> \
      --ood_method MSP,MAXLOGIT,ENERGY,GEN,PRO_GEN \
      --method <FT|ICON> --seed <1..5>
  ```
- **RainbowPrompt extras**: `--method RainbowPrompt --rp_length 20 --rp_D1 56 --rp_D2 28 --lr 0.03 --batch-size 24`. Needs `integrate-RainbowPrompt` branch.
- Do **not** trust hyperparameters from `baseline_commands.md` blindly — always cross-check the paper wandb run config (see `memory/feedback_baseline_cmd_errors.md`).

### Testing Requirements
- No unit tests. Validate via short `--num_tasks 3` runs.
- Reproduction project: `OODVIL/reproduce_check`; naming `<method>_<dataset>_s<seed>`.

### Common Patterns
- Outputs land under `save/<exp>/...` (gitignored).
- wandb entity: `OODVIL`.

## Dependencies

### Internal
- `experiments/baseline_commands.md`, `docs/baselines.md`, `docs/results.md` (outside this repo).

### External
- PyTorch, timm, wandb.
- GitHub remote: `guhyun9454/OOD-VIL` (two active branches).

<!-- MANUAL: -->
