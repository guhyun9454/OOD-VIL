import torch
import numpy as np

# ICON 엔진의 기능을 재사용하기 위해 상속
from engines.ICON import Engine as IconEngine
from engines.ICON import load_model as icon_load_model
from timm.utils import accuracy
import utils


def load_model(args):
    """ICON에서 사용한 ViT backbone을 그대로 활용합니다."""
    return icon_load_model(args)


class Engine(IconEngine):
    """
    Proto-Boundary Learning(PBL) 엔진
    ICON 엔진의 학습·평가 파이프라인을 그대로 활용하면서,
    프로토타입-경계 관리 로직을 확장합니다.
    """

    def __init__(self, model=None, device=None, class_mask=[], domain_list=[], args=None):
        super().__init__(model=model, device=device, class_mask=class_mask, domain_list=domain_list, args=args)
        self.args = args
        # {class_id: [(center, radius), ...]}
        self.prototypes = {}

        # PBL 하이퍼파라미터
        self.tau_split = args.pbl_tau_split
        self.lambda_compact = args.pbl_lambda_compact
        self.lambda_sep = args.pbl_lambda_sep
        self.alpha = args.pbl_alpha

    # ------------ 프로토타입 관리 유틸리티 --------------
    def _init_prototype(self, feature):
        center = feature.detach().clone()
        # learnable radius parameter
        radius = torch.nn.Parameter(torch.tensor(1.0, device=feature.device))
        # 모든 radius 파라미터를 추적하여 별도 최적화
        if not hasattr(self, 'radius_params'):
            self.radius_params = torch.nn.ParameterList()
        self.radius_params.append(radius)
        return [center, radius]

    def _update_prototype(self, proto, feature):
        center, radius = proto
        center.data = (1 - self.alpha) * center.data + self.alpha * feature.detach()
        dist = torch.norm(feature.detach() - center.data, p=2)
        radius.data = (1 - self.alpha) * radius.data + self.alpha * dist
        return [center, radius]

    def _assign_prototype(self, class_id, feature):
        if class_id not in self.prototypes or len(self.prototypes[class_id]) == 0:
            self.prototypes.setdefault(class_id, []).append(self._init_prototype(feature))
            return 0
        dists = [torch.norm(feature - p[0], p=2) for p in self.prototypes[class_id]]
        min_idx = int(torch.argmin(torch.stack(dists)))
        if dists[min_idx] > self.tau_split:
            self.prototypes[class_id].append(self._init_prototype(feature))
            return len(self.prototypes[class_id]) - 1
        else:
            self.prototypes[class_id][min_idx] = self._update_prototype(self.prototypes[class_id][min_idx], feature)
            return min_idx

    # ------------- Loss ----------------
    def _compactness_loss(self, feature, proto):
        center, radius = proto
        dist_sq = torch.norm(feature - center, p=2) ** 2
        return (radius ** 2) * dist_sq + self.lambda_compact * (radius ** 2)

    # ------------- Override train_one_epoch ----------------
    def train_one_epoch(self, model: torch.nn.Module, criterion, data_loader, optimizer, device, epoch: int, max_norm: float = 0,
                        set_training_mode=True, task_id=-1, class_mask=None, ema_model=None, args=None):
        model.train(set_training_mode)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        header = f'PBL Train: Epoch[{epoch+1:{int(np.log10(args.epochs))+1}}/{args.epochs}]'

        for batch_idx, (inputs, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if args.develop and batch_idx > 20:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss_cls = criterion(outputs, targets)

            features = model.forward_features(inputs)[:, 0]
            compact_loss = 0.0
            for feature, label in zip(features, targets):
                proto_idx = self._assign_prototype(label.item(), feature.detach())
                proto = self.prototypes[label.item()][proto_idx]
                compact_loss += self._compactness_loss(feature, proto)
            compact_loss = compact_loss / features.size(0)

            total_loss = loss_cls + self.lambda_compact * compact_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # --- radius 파라미터 업데이트 (simple SGD) ---
            if hasattr(self, 'radius_params') and self.radius_params:
                for r in self.radius_params:
                    if r.grad is not None:
                        r.data -= args.lr * r.grad  # 동일 learning rate 사용
                        r.grad = None

            metric_logger.update(Loss=total_loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inputs.size(0))
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inputs.size(0))

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # ------------- OOD Score --------------
    def compute_ood_score(self, feature):
        """특정 feature에 대한 OOD score를 계산합니다.

        반환값은 **파이썬 float** 이어야 이후 numpy 연산에서 문제가 발생하지 않습니다.
        """

        min_score = float('inf')  # 가장 작은 score (ID일수록 작음)

        for class_id, plist in self.prototypes.items():
            for center, radius in plist:
                # 거리 및 스코어 계산 (tensor scalar)
                dist = torch.norm(feature - center, p=2)
                score = dist / (radius + 1e-6)

                # Tensor → float 변환을 통해 python 영역에서 비교 수행
                score_val = score.item()
                if score_val < min_score:
                    min_score = score_val

        return min_score  # float 값

    # ------------- OOD 평가 --------------
    @torch.no_grad()
    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args, task_id=None):
        """PBL 방식(OOD score = min(dist/r))으로 AUROC 계산"""
        from sklearn.metrics import roc_auc_score
        model.eval()

        # helper to get features & scores
        def _get_scores(dataloader):
            scores = []
            for batch_idx, (imgs, _) in enumerate(dataloader):
                if args.develop and batch_idx > 20:
                    break
                # dataloader가 단일 샘플을 반환할 경우 배치 차원 추가
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(0)
                imgs = imgs.to(device, non_blocking=True)
                feats = model.forward_features(imgs)[:, 0]
                for f in feats:
                    scores.append(self.compute_ood_score(f))
            return scores

        id_loader = id_datasets if isinstance(id_datasets, torch.utils.data.DataLoader) else \
            torch.utils.data.DataLoader(id_datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        if isinstance(ood_dataset, torch.utils.data.DataLoader):
            ood_loader = ood_dataset
        else:
            ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        id_scores = _get_scores(id_loader)
        ood_scores = _get_scores(ood_loader)

        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        auroc = roc_auc_score(labels, -scores)  # score 낮을수록 ID

        print(f"[PBL-OOD] AUROC: {auroc*100:.2f}% (Task {task_id+1 if task_id is not None else '-'})")
        if args.wandb:
            import wandb
            wandb.log({"OOD_AUROC": auroc, "TASK": task_id})
        return auroc 