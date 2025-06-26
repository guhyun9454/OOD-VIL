import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Circle

# ICON 엔진의 기능을 재사용하기 위해 상속
from engines.ICON import Engine as IconEngine
from engines.ICON import load_model as icon_load_model
from timm.utils import accuracy
import utils
from utils import save_anomaly_histogram


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
            print(f"PBL LOG: [Class {class_id}] Initializing first prototype.")
            self.prototypes.setdefault(class_id, []).append(self._init_prototype(feature))
            print(f"PBL LOG: [Class {class_id}] Now has {len(self.prototypes[class_id])} prototypes.")
            return 0
        
        dists = [torch.norm(feature - p[0], p=2) for p in self.prototypes[class_id]]
        min_dist, min_idx = torch.min(torch.stack(dists)), int(torch.argmin(torch.stack(dists)))

        if min_dist > self.tau_split:
            print(f"PBL LOG: [Class {class_id}] Distance {min_dist:.2f} > tau_split({self.tau_split}). Adding new prototype.")
            self.prototypes[class_id].append(self._init_prototype(feature))
            print(f"PBL LOG: [Class {class_id}] Now has {len(self.prototypes[class_id])} prototypes.")
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
        """ICON 의 증분학습 로직을 그대로 가져오고, PBL 전용 compactness loss 만 추가"""

        model.train(set_training_mode)

        # --- PBL LOGGING ---
        if epoch == 0: # Log only on the first epoch of the task
            current_classes = class_mask[task_id]
            try:
                if self.domain_list and len(self.domain_list) > max(current_classes):
                    current_domains = {c: self.domain_list[c] for c in current_classes}
                    print(f"PBL LOG: Training Task {task_id}. Current Classes: {current_classes}. Domains: {current_domains}")
                else:
                    print(f"PBL LOG: Training Task {task_id}. Current Classes: {current_classes}.")
            except Exception as e:
                print(f"PBL LOG: Could not log domain info. Error: {e}")
                print(f"PBL LOG: Training Task {task_id}. Current Classes: {current_classes}.")
        # --- END LOGGING ---

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        header = f'PBL Train: Epoch[{epoch+1:{int(np.log10(args.epochs))+1}}/{args.epochs}]'

        for batch_idx, (inputs, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if args.develop and batch_idx > 20:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # ---------------- ICON forward & loss 구성 -----------------
            outputs = model(inputs)  # (bs, class + n)

            # distillation loss (from ICON)
            distill_loss = 0
            if self.distill_head is not None:
                feature_for_distill = model.forward_features(inputs)[:, 0]
                output_distill = self.distill_head(feature_for_distill)
                mask = torch.isin(torch.tensor(self.labels_in_head), torch.tensor(self.current_classes))
                cur_class_nodes = torch.where(mask)[0]
                m = torch.isin(torch.tensor(self.labels_in_head[cur_class_nodes]), torch.tensor(list(self.added_classes_in_cur_task)))
                distill_node_indices = self.labels_in_head[cur_class_nodes][~m]
                distill_loss = self.kl_div(outputs[:, distill_node_indices], output_distill[:, distill_node_indices])

            # ICON 의 get_max_label_logits 로 multi-node 통합
            if outputs.shape[-1] > self.num_classes:
                outputs, _, _ = self.get_max_label_logits(outputs, class_mask[task_id], slice=False)
                if len(self.added_classes_in_cur_task) > 0:
                    for added_class in self.added_classes_in_cur_task:
                        cur_node = np.where(self.labels_in_head == added_class)[0][-1]
                        outputs[:, added_class] = outputs[:, cur_node]
                outputs = outputs[:, :self.num_classes]

            # 현재 task 가 아닌 클래스는 masking
            if class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = outputs.index_fill(dim=1, index=not_mask, value=float('-inf'))
            else:
                logits = outputs

            # --------------- PBL Compactness loss 계산 -----------------
            features = model.forward_features(inputs)[:, 0]
            compact_loss = 0.0
            for feature, label in zip(features, targets):
                proto_idx = self._assign_prototype(label.item(), feature.detach())
                proto = self.prototypes[label.item()][proto_idx]
                compact_loss += self._compactness_loss(feature, proto)
            compact_loss = compact_loss / features.size(0)

            # --------------- 최종 loss -----------------
            loss = criterion(logits, targets) + distill_loss + self.lambda_compact * compact_loss

            if not torch.isfinite(loss):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # radius 파라미터 업데이트 (simple SGD)
            if hasattr(self, 'radius_params') and self.radius_params:
                for r in self.radius_params:
                    if r.grad is not None:
                        r.data -= args.lr * r.grad
                        r.grad = None

            # ---------------- metric logging ----------------
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inputs.size(0))
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inputs.size(0))

            if ema_model is not None:
                ema_model.update(model.get_adapter())

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # ------------- OOD Score --------------
    def compute_ood_score(self, feature):
        """특정 feature에 대한 OOD score를 계산합니다."""
        min_score = float('inf')
        feature_tensor = torch.from_numpy(feature).to(self.device)

        for class_id, plist in self.prototypes.items():
            for center, radius in plist:
                dist = torch.norm(feature_tensor - center, p=2)
                score = dist / (radius + 1e-6)
                score_val = score.item()
                if score_val < min_score:
                    min_score = score_val

        return min_score

    def _get_features_and_labels(self, model, dataloader, device, is_ood=False):
        """Helper function to extract features and labels from a dataloader."""
        all_features = []
        all_labels = []
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            if batch_idx > 10 and self.args.develop:
                break
            if imgs.dim() == 3: imgs = imgs.unsqueeze(0)
            imgs = imgs.to(device, non_blocking=True)
            feats = model.forward_features(imgs)[:, 0]
            all_features.append(feats.cpu().numpy())
            if is_ood:
                all_labels.append(np.full(len(imgs), -1)) # -1 for OOD
            else:
                all_labels.append(labels.cpu().numpy())
        return np.concatenate(all_features), np.concatenate(all_labels)

    # ------------- OOD 평가 --------------
    @torch.no_grad()
    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args, task_id=None):
        """PBL 방식(OOD score = min(dist/r))으로 AUROC 계산"""
        from sklearn.metrics import roc_auc_score
        model.eval()

        id_loader = id_datasets if isinstance(id_datasets, torch.utils.data.DataLoader) else \
            torch.utils.data.DataLoader(id_datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        ood_loader = ood_dataset if isinstance(ood_dataset, torch.utils.data.DataLoader) else \
            torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # 1. Extract features ONCE
        print("PBL LOG: Extracting features for ID and OOD datasets...")
        id_features, id_labels = self._get_features_and_labels(model, id_loader, device)
        ood_features, ood_labels = self._get_features_and_labels(model, ood_loader, device, is_ood=True)
        print("PBL LOG: Feature extraction complete.")

        # 2. Calculate OOD scores using extracted features
        id_scores = [self.compute_ood_score(f) for f in id_features]
        ood_scores = [self.compute_ood_score(f) for f in ood_features]

        # 3. Visualize using extracted features
        self.visualize_prototypes_tsne(id_features, id_labels, ood_features, task_id, args)

        # Anomaly score 히스토그램 저장 (verbose 모드나 wandb 활성화 시)
        if args.verbose or args.wandb:
            id_scores_np = np.array(id_scores)
            ood_scores_np = np.array(ood_scores)
            
            hist_path = save_anomaly_histogram(
                id_scores_np, 
                ood_scores_np, 
                args, 
                suffix='pbl', 
                task_id=task_id
            )
            
            if args.wandb:
                import wandb
                wandb.log({f"Anomaly Histogram TASK {task_id}": wandb.Image(hist_path)})

        # Binary labels: 1 for ID, 0 for OOD
        binary_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        all_scores = np.concatenate([id_scores, ood_scores])

        # PBL 점수의 경우 값이 작을수록 ID이므로 부호를 반전하여 사용
        from sklearn import metrics
        fpr, tpr, _ = metrics.roc_curve(binary_labels, -all_scores, drop_intermediate=False)
        auroc = metrics.auc(fpr, tpr)

        # FPR@TPR95 계산
        idx_tpr95 = np.abs(tpr - 0.95).argmin()
        fpr_at_tpr95 = fpr[idx_tpr95]

        print(f"[PBL-OOD]: evaluating metrics...")
        print(f"AUROC: {auroc*100:.2f}%, FPR@TPR95: {fpr_at_tpr95*100:.2f}% (Task {task_id+1 if task_id is not None else '-'})")

        # W&B 로깅 (옵션)
        if args.wandb:
            import wandb
            wandb.log({"OOD_AUROC (↑)": auroc*100, "FPR@TPR95 (↓)": fpr_at_tpr95*100, "TASK": task_id})

        return {"auroc": auroc, "fpr_at_tpr95": fpr_at_tpr95, "scores": all_scores}

    def visualize_prototypes_tsne(self, id_features, id_labels, ood_features, task_id, args):
        print("PBL LOG: Starting t-SNE visualization...")
        
        # 1. Create directory for plots
        save_dir = os.path.join(args.save, 'tsne_plots')
        os.makedirs(save_dir, exist_ok=True)

        # 2. Combine features
        all_features = np.concatenate([id_features, ood_features])
        all_labels = np.concatenate([id_labels, np.full(len(ood_features), -1)])

        # 3. Gather prototypes
        proto_centers_list = []
        proto_radii = []
        proto_labels = []
        if self.prototypes:
            for class_id, plist in self.prototypes.items():
                for center, radius in plist:
                    proto_centers_list.append(center.cpu().detach().numpy())
                    proto_radii.append(radius.item())
                    proto_labels.append(class_id)

        if not proto_centers_list:
            print("PBL LOG: No prototypes to visualize. Skipping t-SNE.")
            return
        
        proto_centers = np.array(proto_centers_list)

        # 4. Run t-SNE
        tsne_data = np.vstack([all_features, proto_centers])
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=args.seed)
        tsne_results = tsne.fit_transform(tsne_data)
        
        feat_tsne = tsne_results[:-len(proto_centers)]
        proto_tsne = tsne_results[-len(proto_centers):]

        # 5. Plotting
        plt.figure(figsize=(20, 16))
        ax = plt.gca()
        
        unique_labels = np.unique(all_labels)
        colors = plt.get_cmap('tab20', len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            idx = all_labels == label
            if label == -1:
                ax.scatter(feat_tsne[idx, 0], feat_tsne[idx, 1], c='gray', label='OOD', alpha=0.2, s=10)
            else:
                ax.scatter(feat_tsne[idx, 0], feat_tsne[idx, 1], color=colors(i), label=f'Class {label}', alpha=0.5, s=15)

        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        scaling_factor = x_range / 150.0

        for i in range(len(proto_tsne)):
            center_2d = proto_tsne[i]
            radius = proto_radii[i]
            label = proto_labels[i]
            
            label_idx_list = np.where(unique_labels == label)[0]
            if len(label_idx_list) > 0:
                color = colors(label_idx_list[0])
                ax.scatter(center_2d[0], center_2d[1], c=[color], marker='*', s=300, edgecolor='black', zorder=5)
                circle = Circle(center_2d, radius * scaling_factor, color=color, fill=False, linewidth=2, linestyle='--', alpha=0.8, zorder=4)
                ax.add_patch(circle)

        plt.title(f't-SNE Visualization for Task {task_id}', fontsize=20)
        plt.xlabel('t-SNE dimension 1', fontsize=14)
        plt.ylabel('t-SNE dimension 2', fontsize=14)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        plot_path = os.path.join(save_dir, f'task_{task_id}_tsne.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"PBL LOG: Saved t-SNE visualization to {plot_path}")
