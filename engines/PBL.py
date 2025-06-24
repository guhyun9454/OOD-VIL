import torch
import numpy as np

# ICON 엔진의 기능을 재사용하기 위해 상속
from engines.ICON import Engine as IconEngine
from engines.ICON import load_model as icon_load_model
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
        radius = torch.tensor(1.0, device=feature.device, requires_grad=True)
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

            metric_logger.update(Loss=total_loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inputs.size(0))
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inputs.size(0))

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # ------------- OOD Score --------------
    def compute_ood_score(self, feature):
        min_score = float('inf')
        for class_id, plist in self.prototypes.items():
            for center, radius in plist:
                dist = torch.norm(feature - center, p=2)
                score = dist / (radius + 1e-6)
                if score < min_score:
                    min_score = score
        return min_score

    # TODO: evaluate_ood를 PBL 방식으로 재정의 가능 