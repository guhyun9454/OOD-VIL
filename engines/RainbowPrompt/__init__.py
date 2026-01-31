"""
RainbowPrompt engine (vendored) for OOD-VIL.

Usage:
  python main.py --method RainbowPrompt ...

This package vendors the minimal RainbowPrompt implementation under:
  engines/RainbowPrompt/{vision_transformer.py,prompt.py,attention.py}

No external repo path is required.
No timm registry monkeypatching is used.
"""

from __future__ import annotations

import os
import time
import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
from timm.utils import accuracy
from timm.models import create_model as timm_create_model

from utils import save_accuracy_heatmap, save_anomaly_histogram
from continual_datasets.dataset_utils import RandomSampleWrapper
from OODdetectors.ood_adapter import compute_ood_scores, SUPPORTED_METHODS

from . import vision_transformer as rp_vit


# ---------------------------------------------------------------------------
# Module-level state (kept out of checkpoints)
# ---------------------------------------------------------------------------

_ORIGINAL_MODEL: Optional[torch.nn.Module] = None


def _setdefault(args, key: str, value):
    if not hasattr(args, key) or getattr(args, key) is None:
        setattr(args, key, value)


def _as_list_of_ints(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def load_model(args):
    """
    Build RainbowPrompt prompt model + a frozen original model for cls_features.

    We initialize weights by loading from timm's pretrained ViT (when args.pretrained=True),
    then `strict=False` to ignore prompt-specific parameters.
    """
    global _ORIGINAL_MODEL

    _setdefault(args, "rp_model", "vit_base_patch16_224")
    _setdefault(args, "rp_length", 20)
    _setdefault(args, "rp_pool_size", None)  # default: num_tasks
    _setdefault(args, "rp_top_k", 1)
    _setdefault(args, "rp_balancing", 1.0)
    _setdefault(args, "rp_warm_up", 1)
    _setdefault(args, "rp_relation_type", "attention")
    _setdefault(args, "rp_use_linear", False)
    _setdefault(args, "rp_D1", 1)
    _setdefault(args, "rp_D2", 1)
    _setdefault(args, "rp_KI_iter", 10)
    _setdefault(args, "rp_clip_grad", 1.0)

    e_prompt_layer_idx = _as_list_of_ints(getattr(args, "rp_e_prompt_layer_idx", None))
    if e_prompt_layer_idx is None:
        e_prompt_layer_idx = list(range(12))  # ViT-B/16 depth
    _setdefault(args, "rp_e_prompt_layer_idx", e_prompt_layer_idx)

    self_attn_idx = _as_list_of_ints(getattr(args, "rp_self_attn_idx", None))
    if self_attn_idx is None:
        self_attn_idx = [0, 1, 2, 3, 4, 5]
    _setdefault(args, "rp_self_attn_idx", self_attn_idx)

    _setdefault(args, "rp_prompt_pool", True)
    _setdefault(args, "rp_prompt_key", True)
    _setdefault(args, "rp_batchwise_prompt", False)
    _setdefault(args, "rp_embedding_key", "cls")
    _setdefault(args, "rp_prompt_init", "uniform")
    _setdefault(args, "rp_prompt_key_init", "uniform")
    _setdefault(args, "rp_head_type", "token")
    _setdefault(args, "rp_use_prompt_mask", True)
    _setdefault(args, "rp_same_key_value", False)
    _setdefault(args, "rp_use_e_prompt", True)
    _setdefault(args, "rp_use_prefix_tune_for_e_prompt", True)

    _setdefault(args, "rp_freeze", ["blocks", "patch_embed", "cls_token", "norm", "pos_embed"])

    if str(args.rp_model) != "vit_base_patch16_224":
        raise ValueError(
            "현재 OOD-VIL 내장 RainbowPrompt는 `--rp_model vit_base_patch16_224`만 지원합니다. "
            f"(입력: {args.rp_model})"
        )

    pool_size = int(args.rp_pool_size) if args.rp_pool_size is not None else int(args.num_tasks)

    # 1) Build original model (no prompts) to produce cls_features (pre_logits)
    original_model = rp_vit.vit_base_patch16_224(
        pretrained=False,
        num_classes=int(args.num_classes),
        use_e_prompt=False,
        use_prefix_tune_for_e_prompt=False,
        head_type="token",
    )
    for p in original_model.parameters():
        p.requires_grad = False
    original_model.eval()

    # 2) Build prompt model
    model = rp_vit.vit_base_patch16_224(
        pretrained=False,
        num_classes=int(args.num_classes),
        prompt_length=int(args.rp_length),
        embedding_key=str(args.rp_embedding_key),
        prompt_init=str(args.rp_prompt_init),
        prompt_pool=bool(args.rp_prompt_pool),
        prompt_key=bool(args.rp_prompt_key),
        pool_size=int(pool_size),
        top_k=int(args.rp_top_k),
        batchwise_prompt=bool(args.rp_batchwise_prompt),
        prompt_key_init=str(args.rp_prompt_key_init),
        head_type=str(args.rp_head_type),
        use_prompt_mask=bool(args.rp_use_prompt_mask),
        use_e_prompt=bool(args.rp_use_e_prompt),
        e_prompt_layer_idx=list(args.rp_e_prompt_layer_idx),
        use_prefix_tune_for_e_prompt=bool(args.rp_use_prefix_tune_for_e_prompt),
        same_key_value=bool(args.rp_same_key_value),
        n_tasks=int(args.num_tasks),
        D1=int(args.rp_D1),
        relation_type=str(args.rp_relation_type),
        use_linear=bool(args.rp_use_linear),
        warm_up=int(args.rp_warm_up),
        KI_iter=int(args.rp_KI_iter),
        self_attn_idx=list(args.rp_self_attn_idx),
        D2=int(args.rp_D2),
    )

    # 3) Load pretrained weights from timm backbone (optional)
    if bool(getattr(args, "pretrained", True)):
        base = timm_create_model("vit_base_patch16_224", pretrained=True, num_classes=int(args.num_classes))
        state = base.state_dict()
        original_model.load_state_dict(state, strict=False)
        model.load_state_dict(state, strict=False)

    # 4) Freeze backbone parts (prompt/head remain trainable)
    freeze_prefixes = tuple(str(x) for x in (args.rp_freeze or []))
    if freeze_prefixes:
        for n, p in model.named_parameters():
            if n.startswith(freeze_prefixes):
                p.requires_grad = False

    _ORIGINAL_MODEL = original_model
    return model


class _RainbowPromptLogitsWrapper(torch.nn.Module):
    """
    Adapt RainbowPrompt model (returns dict) -> logits tensor for OOD postprocessors.
    """

    def __init__(self, prompt_model: torch.nn.Module, original_model: torch.nn.Module, task_id: int, learned_id: int):
        super().__init__()
        self.prompt_model = prompt_model
        self.original_model = original_model
        self.task_id = int(task_id)
        self.learned_id = int(learned_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.original_model(x)
            cls_features = out["pre_logits"] if isinstance(out, dict) and "pre_logits" in out else out
        out_p = self.prompt_model(x, task_id=self.task_id, learned_id=self.learned_id, cls_features=cls_features, train=False)
        return out_p["logits"] if isinstance(out_p, dict) and "logits" in out_p else out_p


class Engine:
    def __init__(self, model=None, device=None, class_mask=[], domain_list=[], args=None):
        self.model = model
        self.device = device
        self.args = args
        self.class_mask = class_mask
        self.domain_list = domain_list
        self.num_tasks = args.num_tasks

        if _ORIGINAL_MODEL is None:
            raise RuntimeError("RainbowPrompt original model이 초기화되지 않았습니다. load_model(args) 호출이 필요합니다.")
        self.original_model = _ORIGINAL_MODEL

    def _get_cls_features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        with torch.no_grad():
            out = self.original_model(x)
            if isinstance(out, dict) and "pre_logits" in out:
                return out["pre_logits"]
            if isinstance(out, torch.Tensor):
                return out
        return None

    def _mask_logits_train(self, logits: torch.Tensor, task_id: int, class_mask, args) -> torch.Tensor:
        if class_mask is None:
            return logits
        if not getattr(args, "train_mask", True):
            return logits
        mask = class_mask[task_id]
        not_mask = np.setdiff1d(np.arange(int(args.num_classes)), mask)
        if len(not_mask) == 0:
            return logits
        not_mask_t = torch.tensor(not_mask, dtype=torch.int64, device=logits.device)
        return logits.index_fill(dim=1, index=not_mask_t, value=float("-inf"))

    def train_one_epoch(self, model, criterion, data_loader, optimizer, device, epoch, task_id, class_mask, args):
        model.train()
        self.original_model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if getattr(args, "develop", False) and batch_idx > 20:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            cls_features = self._get_cls_features(inputs)
            out = model(
                inputs,
                task_id=int(task_id),
                learned_id=int(task_id),
                cls_features=cls_features,
                train=True,
                epoch_info=int(epoch),
            )
            logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
            logits = self._mask_logits_train(logits, task_id, class_mask, args)

            loss = criterion(logits, targets)
            if isinstance(out, dict) and "sim_loss" in out:
                loss = loss - float(getattr(args, "rp_balancing", 1.0)) * out["sim_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad = getattr(args, "rp_clip_grad", None)
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            optimizer.step()

            acc1 = accuracy(logits, targets, topk=(1,))[0]
            bs = inputs.size(0)
            total_loss += float(loss.item()) * bs
            total_acc += float(acc1.item()) * bs
            total_samples += bs

            if batch_idx % int(getattr(args, "print_freq", 1000)) == 0:
                cur_avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                cur_avg_acc = total_acc / total_samples if total_samples > 0 else 0.0
                print(
                    f"Task {task_id+1} | Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(data_loader)}]: "
                    f"Loss={loss.item():.4f}, Acc@1={acc1.item():.2f}, "
                    f"Running Avg Loss={cur_avg_loss:.4f}, Running Avg Acc={cur_avg_acc:.2f}"
                )

        epoch_avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_avg_acc = total_acc / total_samples if total_samples > 0 else 0.0
        return epoch_avg_loss, epoch_avg_acc

    @torch.no_grad()
    def evaluate_task(self, model, data_loader, device, task_id, learned_id, class_mask, args):
        model.eval()
        self.original_model.eval()
        criterion = torch.nn.CrossEntropyLoss().to(device)

        total_acc = 0.0
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if getattr(args, "develop", False) and batch_idx > 20:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            cls_features = self._get_cls_features(inputs)
            out = model(inputs, task_id=int(task_id), learned_id=int(learned_id), cls_features=cls_features, train=False)
            logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
            loss = criterion(logits, targets)
            acc1 = accuracy(logits, targets, topk=(1,))[0]

            bs = inputs.size(0)
            total_loss += float(loss.item()) * bs
            total_acc += float(acc1.item()) * bs
            total_samples += bs

        avg_acc = total_acc / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(f"Task {task_id+1}: Final Avg Loss = {avg_loss:.4f} | Final Avg Acc@1 = {avg_acc:.4f}")
        return avg_acc

    def evaluate_till_now(self, model, data_loader, device, task_id, class_mask, acc_matrix, args):
        for t in range(task_id + 1):
            acc_matrix[t, task_id] = self.evaluate_task(
                model=model,
                data_loader=data_loader[t]["val"],
                device=device,
                task_id=t,
                learned_id=task_id,
                class_mask=class_mask,
                args=args,
            )

        A_i = [np.mean(acc_matrix[: i + 1, i]) for i in range(task_id + 1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)

        result_str = "[Average accuracy till task{}] A_last: {:.2f} A_avg: {:.2f}".format(task_id + 1, A_last, A_avg)
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += " Forgetting: {:.4f}".format(forgetting)
        else:
            forgetting = 0

        if getattr(args, "wandb", False):
            import wandb

            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})

        print(result_str)
        if getattr(args, "verbose", False) or getattr(args, "wandb", False):
            sub_matrix = acc_matrix[: task_id + 1, : task_id + 1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            heatmap_path = save_accuracy_heatmap(result, task_id, args)
            if getattr(args, "wandb", False):
                import wandb

                wandb.log({"Accuracy Heatmap": wandb.Image(heatmap_path)})

        return {"Acc@1": A_last}

    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args, task_id=None):
        model.eval()
        self.original_model.eval()

        ood_method = str(getattr(args, "ood_method", "ALL")).upper()

        id_size, ood_size = len(id_datasets), len(ood_dataset)
        min_size = min(id_size, ood_size)
        if getattr(args, "develop", False):
            min_size = 1000
        if getattr(args, "ood_develop", None):
            min_size = int(args.ood_develop)

        if getattr(args, "verbose", False):
            print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")

        id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
        ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset

        id_loader = torch.utils.data.DataLoader(
            id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        if ood_method == "ALL":
            methods = SUPPORTED_METHODS
        else:
            methods = [m.strip().upper() for m in ood_method.split(",") if m.strip()]
            unsupported = [m for m in methods if m not in SUPPORTED_METHODS]
            if unsupported:
                raise ValueError(f"지원되지 않는 OOD 메소드: {unsupported}. 지원되는 메소드: {SUPPORTED_METHODS}")

        cur_task = int(task_id) if task_id is not None else int(getattr(args, "num_tasks", 1)) - 1
        wrapper = _RainbowPromptLogitsWrapper(
            prompt_model=model, original_model=self.original_model, task_id=cur_task, learned_id=cur_task
        )
        wrapper.to(device)

        from sklearn import metrics

        results: Dict[str, Dict[str, Any]] = {}
        for method in methods:
            id_scores, ood_scores = compute_ood_scores(method, wrapper, id_loader, ood_loader, device)

            if getattr(args, "verbose", False) or getattr(args, "wandb", False):
                hist_path = save_anomaly_histogram(id_scores.numpy(), ood_scores.numpy(), args, suffix=method.lower(), task_id=task_id)
                if getattr(args, "wandb", False):
                    import wandb

                    wandb.log({f"Anomaly Histogram TASK {task_id}": wandb.Image(hist_path)})

            binary_labels = np.concatenate([np.ones(id_scores.shape[0]), np.zeros(ood_scores.shape[0])])
            all_scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])

            fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
            auroc = metrics.auc(fpr, tpr)
            idx_tpr95 = np.abs(tpr - 0.95).argmin()
            fpr_at_tpr95 = fpr[idx_tpr95]

            print(f"[{method}]: AUROC {auroc * 100:.2f}% | FPR@TPR95 {fpr_at_tpr95 * 100:.2f}%")
            if getattr(args, "wandb", False):
                import wandb

                wandb.log(
                    {f"{method}_AUROC (↑)": auroc * 100, f"{method}_FPR@TPR95 (↓)": fpr_at_tpr95 * 100, "TASK": task_id}
                )

            results[method] = {"auroc": auroc, "fpr_at_tpr95": fpr_at_tpr95, "scores": all_scores}

        return results

    def train_and_evaluate(self, model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, args):
        self.original_model.to(device)
        self.original_model.eval()

        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            print(f"{f'Training on Task {task_id+1}/{args.num_tasks} (RainbowPrompt)':=^60}")
            train_start = time.time()

            for epoch in range(args.epochs):
                epoch_start = time.time()
                epoch_avg_loss, epoch_avg_acc = self.train_one_epoch(
                    model=model,
                    criterion=criterion,
                    data_loader=data_loader[task_id]["train"],
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    task_id=task_id,
                    class_mask=class_mask,
                    args=args,
                )
                epoch_duration = time.time() - epoch_start
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Completed in {str(datetime.timedelta(seconds=int(epoch_duration)))}: "
                    f"Avg Loss = {epoch_avg_loss:.4f}, Avg Acc@1 = {epoch_avg_acc:.2f}"
                )
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch)

            train_duration = time.time() - train_start
            print(f"Task {task_id+1} training completed in {str(datetime.timedelta(seconds=int(train_duration)))}")

            print(f"{f'Testing on Task {task_id+1}/{args.num_tasks}':=^60}")
            eval_start = time.time()
            self.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args)
            eval_duration = time.time() - eval_start
            print(f"Task {task_id+1} evaluation completed in {str(datetime.timedelta(seconds=int(eval_duration)))}")

            if getattr(args, "ood_dataset", None):
                print(f"{'OOD Evaluation':=^60}")
                ood_start = time.time()
                all_id_datasets = torch.utils.data.ConcatDataset([data_loader[t]["val"].dataset for t in range(task_id + 1)])
                ood_loader = data_loader[-1]["ood"]
                self.evaluate_ood(model, all_id_datasets, ood_loader, device, args, task_id)
                ood_duration = time.time() - ood_start
                print(f"OOD evaluation after Task {task_id+1} completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")

            if getattr(args, "save", None):
                checkpoint_dir = os.path.join(args.save, "checkpoint")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"task{task_id+1}_checkpoint.pth")
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "method": "RainbowPrompt",
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint for task {task_id+1} at {checkpoint_path}")

    def load_checkpoint(self, model, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print(f"체크포인트를 로드합니다: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        return model

