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

from timm.utils import accuracy
from timm.models import create_model
import utils


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
        """Vectorised compactness loss over a mini-batch."""
        batch_loss = []
        lambda_r = self.args.pbl_lambda_r
        for f, y in zip(feats, targets):
            center, radius = self._update_or_create_proto(int(y.item()), f)
            dist_sq = self._euclidean_squared(f, center)
            comp = (dist_sq / (radius ** 2)) + lambda_r * (radius ** 2)
            batch_loss.append(comp)
        if len(batch_loss) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(batch_loss).mean()

    def train_one_epoch(self, criterion, data_loader, optimizer, epoch):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"PBL Train: Epoch[{epoch+1}/{self.args.epochs}]"

        for samples, targets in metric_logger.log_every(data_loader, self.args.print_freq, header):
            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(samples)
            ce_loss = criterion(logits, targets)

            # feature extraction for PBL
            with torch.no_grad():
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
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "PBL Eval:"
        for samples, targets in metric_logger.log_every(data_loader, self.args.print_freq, header):
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
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # --------------------------------------------------
    #   Public API expected by main.py
    # --------------------------------------------------
    def train_and_evaluate(self, model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, args):
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            print(f"\n{'='*20}  Task {task_id+1}/{args.num_tasks}  {'='*20}")
            for epoch in range(args.epochs):
                self.train_one_epoch(criterion, data_loader[task_id]['train'], optimizer, epoch)
                if lr_scheduler:
                    lr_scheduler.step(epoch)
            # evaluate current and previous tasks
            for i in range(task_id+1):
                stats = self.evaluate(data_loader[i]['val'])
                acc_matrix[i, task_id] = stats['Acc@1']
            # simple feedback
            A_last = acc_matrix[:task_id+1, task_id].mean()
            print(f"[Average accuracy till task{task_id+1}] A_last: {A_last:.2f}")

    def evaluate_till_now(self, *args, **kwargs):
        # Kept for API compatibility – not used because train_and_evaluate already does so.
        pass

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
    @torch.no_grad()
    def evaluate_ood(self, model, id_datasets, ood_loader, device, args, task_id=None):
        model.eval()
        print("Running simple OOD evaluation (MSP)...")
        id_loader = torch.utils.data.DataLoader(id_datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        def gather_scores(d_loader):
            scores = []
            for x, _ in d_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                softmax = F.softmax(logits, dim=1)
                msp = -torch.max(softmax, dim=1)[0]  # negative max prob as OOD score
                scores.append(msp.detach().cpu())
            return torch.cat(scores)

        id_scores = gather_scores(id_loader)
        ood_scores = gather_scores(ood_loader)
        # AUROC
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        preds = np.concatenate([id_scores.numpy(), ood_scores.numpy()])
        auroc = roc_auc_score(labels, preds)
        print(f"OOD AUROC: {auroc*100:.2f}%")
        return auroc 