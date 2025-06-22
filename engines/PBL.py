import math
import sys
import os
import time
import datetime
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import wandb

from timm.utils import accuracy
from timm.models import create_model
import utils
from utils import save_accuracy_heatmap, save_anomaly_histogram


# -----------------------------
#   Utility functions
# -----------------------------

def load_model(args):
    """Create backbone network using timm and freeze all parameters except the classification head."""
    if args.model is None:
        backbone_name = "vit_base_patch16_224"
    else:
        backbone_name = args.model

    model = create_model(
        backbone_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    )

    # Linear probing option (freeze backbone)
    if args.linear_probing:
        for n, p in model.named_parameters():
            if 'head' not in n:
                p.requires_grad = False

    return model


class Engine:
    """Proto-Boundary Learning Engine.

    This is a light-weight implementation that satisfies the required public
    interface so that the training script (main.py) can run without modification.
    It follows the PBL formulation at a high-level, but is kept intentionally
    simple for demonstration purposes. You can enrich any individual component
    later without touching the surrounding code base.
    """

    def __init__(self, model=None, device=None, class_mask=None, domain_list=None, args=None):
        self.model = model
        self.device = device
        self.args = args
        self.class_mask = class_mask if class_mask is not None else []
        self.domain_list = domain_list if domain_list is not None else []
        # Prototype container  ->  {class_idx: List[Tuple[center (Tensor), radius (Tensor)] ] }
        self.prototypes: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.ema_alpha = 0.9  # hard-coded EMA factor for prototype update
        
        self.use_wandb = args.wandb
        
    def print_prototype_status(self, prefix=""):
        if not self.prototypes:
            print(f"{prefix}프로토타입 없음")
            return
        
        total = sum(len(protos) for protos in self.prototypes.values())
        class_counts = {cls: len(protos) for cls, protos in self.prototypes.items()}
        print(f"{prefix}프로토타입 상태: 총 {total}개, 클래스별 = {class_counts}")
        
    # --------------------------------------------------
    #   Private helpers
    # --------------------------------------------------
    @staticmethod
    def _euclidean_squared(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum((x - y) ** 2, dim=-1)

    def _find_nearest_prototype(self, cls: int, feat: torch.Tensor):
        """Return (proto_idx, distance, radius). If no proto exists, returns (None, inf, None)"""
        if cls not in self.prototypes or len(self.prototypes[cls]) == 0:
            return None, torch.tensor(float('inf'), device=feat.device), None
        dists = torch.stack([self._euclidean_squared(feat, c) for c, _ in self.prototypes[cls]])
        min_idx = torch.argmin(dists)
        center, radius = self.prototypes[cls][min_idx]
        return min_idx, dists[min_idx], radius

    def _update_or_create_proto(self, cls: int, feat: torch.Tensor):
        """Split / update logic following a simplified τ_split criterion."""
        tau = self.args.pbl_tau_split
        proto_idx, dist, radius = self._find_nearest_prototype(cls, feat)

        if proto_idx is None or torch.sqrt(dist) > tau:  # split: create new prototype
            center = feat.detach()
            radius = torch.tensor([tau], device=feat.device).detach()
            self.prototypes.setdefault(cls, []).append((center, radius))
            print(f"새 프로토타입 생성: 클래스 {cls} (현재 {len(self.prototypes[cls])}개)")
            return center, radius
        else:
            # EMA update existing prototype
            center, radius = self.prototypes[cls][proto_idx]
            new_center = self.ema_alpha * center + (1 - self.ema_alpha) * feat.detach()
            self.prototypes[cls][proto_idx] = (new_center, radius)
            return new_center, radius

    # --------------------------------------------------
    #   Training / evaluation routines
    # --------------------------------------------------
    def _compute_compactness_loss(self, feats: torch.Tensor, targets: torch.Tensor):
        """Compactness loss (batch-level prototype update).

        1) 배치에 등장한 각 클래스를 위한 평균 피처를 계산해 한 번만 프로토타입을 생성/업데이트합니다.
        2) 업데이트된 프로토타입을 기준으로 각 샘플의 compactness 손실을 계산합니다.
        """

        lambda_r = self.args.pbl_lambda_r

        # -------------------------------
        # 1) 클래스별 평균 피처로 프로토타입 업데이트
        # -------------------------------
        unique_classes = torch.unique(targets)
        proto_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for cls in unique_classes:
            cls_int = int(cls.item())
            cls_feats = feats[targets == cls]  # (N_c, D)
            if cls_feats.numel() == 0:
                continue  # 안전 장치
            mean_feat = cls_feats.mean(dim=0)  # (D,)

            # 한 번만 프로토타입 생성/업데이트
            center, radius = self._update_or_create_proto(cls_int, mean_feat)
            proto_cache[cls_int] = (center, radius)

        if feats.size(0) == 0:
            return torch.tensor(0.0, device=self.device)

        # -------------------------------
        # 2) 각 샘플에 대한 compactness 손실 계산
        # -------------------------------
        centers = torch.stack([proto_cache[int(y.item())][0] for y in targets])  # (B, D)
        radii = torch.stack([proto_cache[int(y.item())][1] for y in targets]).view(-1)  # (B,)

        dist_sq = torch.sum((feats - centers) ** 2, dim=1)  # (B,)
        comp = dist_sq / (radii ** 2) + lambda_r * (radii ** 2)

        return comp.mean()

    def train_one_epoch(self, criterion, data_loader, optimizer, epoch):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"PBL Train: Epoch[{epoch+1}/{self.args.epochs}]"

        for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, self.args.print_freq, header)):
            if self.args.develop and batch_idx > 20:
                break
                    
            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(samples)
            ce_loss = criterion(logits, targets)

            # feature extraction for PBL (allow gradients to flow)
            feats = self.model.forward_features(samples)[:, 0]  # CLS token
            comp_loss = self._compute_compactness_loss(feats, targets)

            loss = ce_loss + self.args.pbl_lambda_comp * comp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            metric_logger.update(Loss=loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=samples.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=samples.shape[0])
            
            # wandb 로깅
            if self.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/ce_loss': ce_loss.item(),
                    'train/compactness_loss': comp_loss.item(),
                    'train/acc1': acc1.item(),
                    'train/acc5': acc5.item(),
                    'train/epoch': epoch,
                    'train/num_prototypes': sum(len(protos) for protos in self.prototypes.values())
                })
                
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        
        # 에포크 끝에 프로토타입 상태 출력
        self.print_prototype_status(f"[Epoch {epoch+1} 완료] ")

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "PBL Eval:"
        for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, self.args.print_freq, header)):
            if self.args.develop and batch_idx > 20:
                break
                    
            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(samples)
            loss = criterion(logits, targets)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            metric_logger.update(Loss=loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=samples.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=samples.shape[0])
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
            top1=metric_logger.meters['Acc@1'],
            top5=metric_logger.meters['Acc@5'],
            losses=metric_logger.meters['Loss']))
            
        # wandb 로깅
        if self.use_wandb:
            wandb.log({
                'eval/loss': metric_logger.meters['Loss'].global_avg,
                'eval/acc1': metric_logger.meters['Acc@1'].global_avg,
                'eval/acc5': metric_logger.meters['Acc@5'].global_avg
            })
            
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # --------------------------------------------------
    #   Public API expected by main.py
    # --------------------------------------------------
    def train_and_evaluate(self, model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, args):
        """Train sequential tasks and measure continual-learning & OOD metrics."""
        
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            print(f"\n{'='*20}  Task {task_id+1}/{args.num_tasks}  {'='*20}")
            for epoch in range(args.epochs):
                self.train_one_epoch(criterion, data_loader[task_id]['train'], optimizer, epoch)
                if lr_scheduler:
                    lr_scheduler.step(epoch)

            # --- Classification evaluation (till now) ----------------
            self.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args)

            # --- OOD evaluation --------------------------------------
            if args.ood_dataset:
                print(f"{'OOD Evaluation':=^60}")
                # concat ID datasets up to current task
                all_id = torch.utils.data.ConcatDataset([dl['val'].dataset for dl in data_loader[:task_id+1]])
                ood_loader = data_loader[-1]['ood']
                self.evaluate_ood(model, all_id, ood_loader, device, args, task_id)
        
        return acc_matrix

    def evaluate_till_now(self, model, data_loader, device, task_id, class_mask, acc_matrix, args):
        """Evaluate all tasks up to current and print A_last, A_avg, Forgetting."""
        for i in range(task_id + 1):
            stats = self.evaluate(data_loader[i]['val'])
            acc_matrix[i, task_id] = stats['Acc@1']

        A_i = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)

        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
        else:
            forgetting = 0.0

        print(f"[Average accuracy till task{task_id+1}] A_last: {A_last:.2f} A_avg: {A_avg:.2f} Forgetting: {forgetting:.2f}")

        # Task 완료 후 프로토타입 상태 출력
        self.print_prototype_status(f"[Task {task_id+1} 완료] ")

        # ICON과 동일한 형식으로 wandb 로깅
        if args.wandb:
            import wandb
            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})
        
        # 정확도 히트맵 생성 및 로깅
        if args.verbose or args.wandb:
            sub_matrix = acc_matrix[:task_id+1, :task_id+1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            heatmap_path = save_accuracy_heatmap(result, task_id, args)
            if args.wandb:
                import wandb
                wandb.log({"Accuracy Heatmap": wandb.Image(heatmap_path)})
        
        return stats

    # checkpoint helpers -------------------------------------------------
    def restore_head_from_checkpoint(self, model, checkpoint):
        return model  # placeholder – PBL keeps head unchanged

    def load_checkpoint(self, model, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
        return model

    # -----------------  OOD evaluation (simple)  -----------------------
    def evaluate_ood(self, model, id_datasets, ood_loader, device, args, task_id=None):
        """Compute OOD metrics (AUROC & FPR@95) using prototype-normalised distance score."""
        model.eval()

        ood_loader = torch.utils.data.DataLoader(
            ood_loader,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Pre-gather prototype tensors for vectorised distance computation
        centers, radii = [], []
        for cls in self.prototypes:
            for c, r in self.prototypes[cls]:
                centers.append(c)
                radii.append(r)
        if len(centers) == 0:
            print("[PBL] No prototypes available for OOD evaluation.")
            return None

        centers = torch.stack(centers).to(device)  # (N, D)
        radii = torch.stack(radii).view(-1).to(device)  # (N,)

        def ood_score(feats: torch.Tensor):
            # feats: (B, D)  return score (B,)
            dist_sq = torch.cdist(feats, centers, p=2) ** 2  # (B, N)
            norm_dist = dist_sq / (radii ** 2).unsqueeze(0)
            score, _ = torch.min(norm_dist, dim=1)
            return score

        def gather_scores(loader):
            res = []
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                
                # 배치 차원이 없는 경우에만 차원 추가
                if x.dim() == 3:  # (C, H, W) 형태인 경우에만
                    x = x.unsqueeze(0)  # (1, C, H, W)로 배치 차원 추가
                
                with torch.no_grad():
                    feats = model.forward_features(x)[:, 0]
                res.append(ood_score(feats).cpu())
            return torch.cat(res)

        # develop 모드일 때 샘플 수 제한
        if args.develop:
            min_size = 1000
            if len(id_datasets) > min_size:
                indices = torch.randperm(len(id_datasets))[:min_size]
                id_datasets = torch.utils.data.Subset(id_datasets, indices)
            if len(ood_loader.dataset) > min_size:
                indices = torch.randperm(len(ood_loader.dataset))[:min_size] 
                ood_subset = torch.utils.data.Subset(ood_loader.dataset, indices)
                ood_loader = torch.utils.data.DataLoader(ood_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        id_loader = torch.utils.data.DataLoader(id_datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        id_scores = gather_scores(id_loader)
        ood_scores = gather_scores(ood_loader)

        # AUROC
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        preds = np.concatenate([id_scores.numpy(), ood_scores.numpy()])
        auroc = roc_auc_score(labels, preds)

        # FPR@95
        id_scores_np = id_scores.numpy()
        ood_scores_np = ood_scores.numpy()
        threshold = np.percentile(id_scores_np, 95)  # 95% ID accepted (TPR=0.95)
        fpr95 = np.mean(ood_scores_np <= threshold)

        print(f"AUROC (↑) : {auroc*100:.2f}%  |  FPR95 (↓) : {fpr95*100:.2f}%")

        # 이상 점수 히스토그램 저장 (verbose 모드 또는 wandb 사용 시)
        if args.verbose or args.wandb:
            hist_path = save_anomaly_histogram(id_scores_np, ood_scores_np, args, suffix='PBL', task_id=task_id)
            if args.wandb:
                import wandb
                wandb.log({f"Anomaly Histogram TASK {task_id}": wandb.Image(hist_path)})

        # ICON과 동일한 형식으로 wandb 로깅
        if args.wandb:
            import wandb
            wandb.log({f"PBL_AUROC (↑)": auroc * 100, f"PBL_FPR@TPR95 (↓)": fpr95 * 100, "TASK": task_id})

        return auroc, fpr95 