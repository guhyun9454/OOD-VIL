import math
import sys
import os
import time
import datetime
import json
from typing import Iterable
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.utils.model_ema import ModelEmaV2
import copy
import utils
import torch.nn.functional as F
from sklearn.cluster import KMeans
from timm.models import create_model
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from continual_datasets.dataset_utils import RandomSampleWrapper
from utils import save_accuracy_heatmap, save_logits_statistics, save_anomaly_histogram
from OODdetectors.ood_adapter import compute_ood_scores, SUPPORTED_METHODS

# -------------------------------------------------------------
# Simple MLP classifier for Task-specific OOD detection (ID vs pOOD)
# -------------------------------------------------------------


class MLPClassifier(nn.Module):
    """간단한 다층 perceptron(ID:0 vs OOD:1)"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        layers = []
        dim_in = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.ReLU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, 2))  # 2-way (ID / Not-ID)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_model(args):
    if args.dataset == 'CORe50':
        init_ICON_CORe50_args(args)
        print("CORe50 dataset args loaded")
    elif args.dataset == 'iDigits':
        init_ICON_iDigits_args(args)
        print("iDigits dataset args loaded")
    elif args.dataset == 'DomainNet':
        init_ICON_DomainNet_args(args)
        print("DomainNet dataset args loaded")
    else:
        init_ICON_default_args(args)
        print("Default dataset args loaded")

    model = create_model(
        "vit_base_patch16_224_ICON",
        pretrained=True,
        num_classes=args.num_classes,  #10
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
        adapt_blocks=args.adapt_blocks,  #[0, 1, 2, 3, 4]
    )
    for n, p in model.named_parameters():
        p.requires_grad = False
        if 'adapter' in n:
            p.requires_grad = True
        if 'head' in n:
            p.requires_grad = True
    return model

class Engine():
    def __init__(self, model=None, device=None, class_mask=[], domain_list=[], args=None):
        self.current_task = 0
        self.current_classes = []
        #! distillation
        self.class_group_num = 5
        self.classifier_pool = [None for _ in range(self.class_group_num)]
        self.class_group_train_count = [0 for _ in range(self.class_group_num)]
        
        self.task_num = len(class_mask)
        self.class_group_size = len(class_mask[0])
        self.distill_head = None
        self.model = model
        
        self.num_classes = max([item for mask in class_mask for item in mask]) + 1
        self.labels_in_head = np.arange(self.num_classes)
        self.added_classes_in_cur_task = set()
        self.head_timestamps = np.zeros_like(self.labels_in_head)
        self.args = args
        
        self.class_mask = class_mask
        self.domain_list = domain_list

        self.task_type = "initial"
        self.args = args
        
        self.adapter_vec = []
        self.task_type_list = []
        self.class_group_list = []
        self.adapter_vec_label = []
        self.device = device
        
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((self.args.num_classes, self.args.num_domains))
            self.label_train_count = np.zeros((self.args.num_classes))
            self.tanh = torch.nn.Tanh()
            
        self.cs = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def kl_div(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        kl = torch.mean(torch.sum(p * torch.log(p / q), dim=1))
        return kl
  
    def set_new_head(self, model, labels_to_be_added, task_id):
        len_new_nodes = len(labels_to_be_added)
        self.labels_in_head = np.concatenate((self.labels_in_head, labels_to_be_added))
        self.added_classes_in_cur_task.update(labels_to_be_added)
        self.head_timestamps = np.concatenate((self.head_timestamps, [task_id] * len_new_nodes))
        prev_weight, prev_bias = model.head.weight, model.head.bias
        prev_shape = prev_weight.shape  # (class, dim)
        new_head = torch.nn.Linear(prev_shape[-1], prev_shape[0] + len_new_nodes)
    
        new_head.weight[:prev_weight.shape[0]].data.copy_(prev_weight)
        new_head.weight[prev_weight.shape[0]:].data.copy_(prev_weight[labels_to_be_added])
        new_head.bias[:prev_weight.shape[0]].data.copy_(prev_bias)
        new_head.bias[prev_weight.shape[0]:].data.copy_(prev_bias[labels_to_be_added])
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added})")
        return new_head
    
    def inference_acc(self, model, data_loader, device):
        print("Start detecting labels to be added...")
        accuracy_per_label = []
        correct_pred_per_label = [0 for i in range(len(self.current_classes))]
        num_instance_per_label = [0 for i in range(len(self.current_classes))]
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if self.args.develop:
                    if batch_idx > 200:
                        break
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                
                if output.shape[-1] > self.num_classes:  # there are already added nodes till now
                    output, _, _ = self.get_max_label_logits(output, self.current_classes)  # get maximum value for each label
                mask = self.current_classes
                not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
                _, pred = torch.max(logits, 1)
                
                correct_predictions = (pred == target)
                for i, label in enumerate(self.current_classes):
                    mask = (target == label)
                    num_correct_pred = torch.sum(correct_predictions[mask])
                    correct_pred_per_label[i] += num_correct_pred.item()
                    num_instance_per_label[i] += sum(mask).item()
        for correct, num in zip(correct_pred_per_label, num_instance_per_label):
            accuracy_per_label.append(round(correct / num, 2))
        return accuracy_per_label
    
    def detect_labels_to_be_added(self, inference_acc, thresholds=[]):
        labels_with_low_accuracy = []
        
        if self.args.d_threshold:
            for label, acc, thre in zip(self.current_classes, inference_acc, thresholds):
                if acc <= thre:
                    labels_with_low_accuracy.append(label)
        else:  # static threshold
            for label, acc in zip(self.current_classes, inference_acc):
                if acc <= self.args.thre:
                    labels_with_low_accuracy.append(label)
                
        print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
        return labels_with_low_accuracy
    
    def find_same_cluster_items(self, vec):
        if self.kmeans.n_clusters == 1:
            other_cluster_vecs = self.adapter_vec_array
            other_cluster_vecs = torch.tensor(other_cluster_vecs, dtype=torch.float32).to(self.device)
            same_cluster_vecs = None
        else:
            predicted_cluster = self.kmeans.predict(vec.unsqueeze(0).detach().cpu())[0]
            same_cluster_vecs = self.adapter_vec_array[self.cluster_assignments == predicted_cluster]
            other_cluster_vecs = self.adapter_vec_array[self.cluster_assignments != predicted_cluster]
            same_cluster_vecs = torch.tensor(same_cluster_vecs, dtype=torch.float32).to(self.device)
            other_cluster_vecs = torch.tensor(other_cluster_vecs, dtype=torch.float32).to(self.device)
        return same_cluster_vecs, other_cluster_vecs
    
    def calculate_l2_distance(self, diff_adapter, other):
        weights = []
        for o in other:
            l2_distance = torch.norm(diff_adapter - o, p=2)
            weights.append(l2_distance.item())
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights)  # normalization
        return weights

    # ------------------------------------------------------------------
    # Task-specific OOD classifier (Logistic Regression) 관련 유틸리티
    # ------------------------------------------------------------------

    def _extract_cls_features(self, model, loader, device):
        """ViT CLS 토큰 특징 추출"""
        feats = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                z = model.forward_features(x)[:, 0]
                feats.append(z.cpu())
        return torch.cat(feats).numpy()

    def _targeted_fgsm(self, model, x, target, eps=0.03):
        """Targeted FGSM 공격으로 pseudo-OOD 샘플 생성"""
        x_adv = x.clone().detach().requires_grad_(True)
        out = model(x_adv)
        loss = F.cross_entropy(out, target)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv - eps * x_adv.grad.sign()
        return torch.clamp(x_adv.detach(), 0, 1)

    # ------------------------------------------------------------------
    #  New: 다양한 pseudo-OOD 생성 방법
    # ------------------------------------------------------------------
    def _mixup(self, x, alpha=1.0):
        """Batch 내부에서 Mixup 으로 생성"""
        lam = np.random.beta(alpha, alpha)
        perm = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1 - lam) * x[perm]
        return torch.clamp(x_mix, 0, 1)

    def _gaussian_noise(self, x, sigma=0.1):
        noise = torch.randn_like(x) * sigma
        return torch.clamp(x + noise, 0, 1)

    def _pgd(self, model, x, target, eps=0.03, alpha=0.01, steps=5):
        """Targeted PGD 공격"""
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv.requires_grad_(True)
        for _ in range(steps):
            out = model(x_adv)
            loss = F.cross_entropy(out, target)
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()
            x_adv.data = x_adv.data - alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv.data - x, min=-eps, max=eps)
            x_adv.data = torch.clamp(x + delta, 0, 1)
        return x_adv.detach()

    def _cutout(self, x, ratio=0.5):
        """임의의 사각형 영역을 0으로 마스킹"""
        b, c, h, w = x.size()
        cut_h, cut_w = int(h * ratio), int(w * ratio)
        cx = torch.randint(0, h, (b,), device=x.device)
        cy = torch.randint(0, w, (b,), device=x.device)
        x_out = x.clone()
        for i in range(b):
            x1 = torch.clamp(cx[i] - cut_h // 2, 0, h)
            y1 = torch.clamp(cy[i] - cut_w // 2, 0, w)
            x2 = torch.clamp(cx[i] + cut_h // 2, 0, h)
            y2 = torch.clamp(cy[i] + cut_w // 2, 0, w)
            x_out[i, :, x1:x2, y1:y2] = 0
        return x_out

    def _colorjitter(self, x, brightness=0.4):
        """간단한 밝기 변조"""
        factor = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(1-brightness, 1+brightness)
        x_j = x * factor
        return torch.clamp(x_j, 0, 1)

    _SUPPORTED_POOD = [
        "FGSM", "PGD", "MIXUP", "GAUSSIAN", "CUTOUT", "COLORJITTER"
    ]

    def train_task_ood_classifier(self, model, data_loader, device, args, task_id):
        """현재 태스크의 ID 데이터를 이용해 Task-specific OOD 분류기를 학습 및 저장"""
        if task_id == 0:
            print(f"[Task {task_id + 1}] 이전 태스크가 없어 OOD 분류기를 학습하지 않습니다.")
            return

        # 저장 디렉터리 미리 정의 (다른 로깅에서도 사용)
        save_dir = Path(args.save) / "task_clf"
        save_dir.mkdir(exist_ok=True, parents=True)

        print(f"[Task {task_id + 1}] ID-vs-NotID 구분기 학습 시작")

        id_loader = data_loader[task_id]['train']
        X_id = self._extract_cls_features(model, id_loader, device)
        y_id = np.zeros(len(X_id))

        prev_classes = np.concatenate(self.class_mask[:task_id])
        id_imgs, id_logits = [], []           # original ID samples & logits (for visualization)
        pseudo_imgs, pseudo_logits = [], []   # raw adv images & logits for logging
        X_pood = []

        # 선택된 pseudo-OOD 생성 기법 목록
        pood_methods = [m.strip().upper() for m in getattr(args, 'pood_methods', 'fgsm').split(',')]
        for m in pood_methods:
            if m not in self._SUPPORTED_POOD:
                raise ValueError(f"지원되지 않는 Pseudo-OOD 방법: {m}. 사용 가능: {self._SUPPORTED_POOD}")

        for x, _ in id_loader:
            x = x.to(device)
            # --- ID 로짓 저장 (1회만) ---
            with torch.no_grad():
                logits_orig = model(x)

            # target_labels: FGSM 용 (다른 방법도 동일하게 사용)
            tgt_indices = torch.randint(0, len(prev_classes), (x.size(0),), device=device)
            target_labels = torch.tensor(prev_classes[tgt_indices.cpu()], dtype=torch.long, device=device)

            for m in pood_methods:
                # --- pseudo 이미지 생성 ---
                if m == "FGSM":
                    x_pood = self._targeted_fgsm(model, x, target_labels)
                elif m == "PGD":
                    x_pood = self._pgd(model, x, target_labels)
                else:
                    gen_fn = getattr(self, f"_{m.lower()}")
                    x_pood = gen_fn(x)

                with torch.no_grad():
                    logits_pood = model(x_pood)

                # FGSM 은 목표 레이블 성공 샘플만 사용, 나머지는 전체 사용
                if m == "FGSM":
                    sel = logits_pood.argmax(1) == target_labels
                else:
                    sel = torch.ones(x_pood.size(0), dtype=torch.bool, device=x.device)

                if sel.any():
                    id_selected = x[sel].detach().cpu()
                    id_logits_sel = logits_orig[sel].detach().cpu()
                    id_imgs.append(id_selected)
                    id_logits.append(id_logits_sel)

                    sel_imgs = x_pood[sel].detach().cpu()
                    sel_logits = logits_pood[sel].detach().cpu()
                    z_adv = model.forward_features(sel_imgs.to(device))[:, 0].cpu()

                    X_pood.append(z_adv)
                    pseudo_imgs.append(sel_imgs)
                    pseudo_logits.append(sel_logits)

        # --- Save & log pseudo OOD information --------------------------------
        if pseudo_imgs:
            pseudo_imgs_tensor = torch.cat(pseudo_imgs)
            pseudo_logits_tensor = torch.cat(pseudo_logits)

            # --- 원본 ID 텐서도 연결 ---
            id_imgs_tensor = torch.cat(id_imgs)
            id_logits_tensor = torch.cat(id_logits)

            # -------- 시각화 & 저장 --------
            import matplotlib.pyplot as plt
            import math

            num_examples = min(8, id_imgs_tensor.size(0))
            fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 2, 4))
            for idx in range(num_examples):
                # ID 이미지
                img_id = id_imgs_tensor[idx]
                img_id_np = img_id.permute(1, 2, 0).numpy()
                img_id_np = np.clip(img_id_np, 0, 1)
                axes[0, idx].imshow(img_id_np)
                axes[0, idx].axis('off')
                id_pred = int(id_logits_tensor[idx].argmax().item())
                axes[0, idx].set_title(f"ID | pred:{id_pred}\nmax: {id_logits_tensor[idx].max():.2f}")

                # pOOD 이미지
                img_pood = pseudo_imgs_tensor[idx]
                img_pood_np = img_pood.permute(1, 2, 0).numpy()
                img_pood_np = np.clip(img_pood_np, 0, 1)
                axes[1, idx].imshow(img_pood_np)
                axes[1, idx].axis('off')
                pood_pred = int(pseudo_logits_tensor[idx].argmax().item())
                axes[1, idx].set_title(f"pOOD | pred:{pood_pred}\nmax: {pseudo_logits_tensor[idx].max():.2f}")

            plt.tight_layout()
            fig_path = save_dir / f"pseudo_ood_comparison_task{task_id + 1}.png"
            plt.savefig(fig_path)
            plt.close(fig)

            # -------- wandb 로깅 --------
            if args.wandb:
                import wandb
                num_preview = min(32, pseudo_imgs_tensor.size(0))
                preview_imgs = [wandb.Image(img) for img in pseudo_imgs_tensor[:num_preview]]
                wandb.log({
                    f"Task{task_id}_PseudoOOD_Comparison": wandb.Image(str(fig_path))
                })

        if not X_pood:
            print("경고: Pseudo-OOD 샘플을 생성하지 못했습니다. 분류기 학습을 건너뜁니다.")
            return

        X_pood = torch.cat(X_pood).numpy()
        y_pood = np.ones(len(X_pood))

        X = np.concatenate([X_id, X_pood])
        y = np.concatenate([y_id, y_pood])

        # -----------------------------
        #  MLP 분류기 학습 (PyTorch)
        # -----------------------------
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        # 하이퍼파라미터: args 에서 가져오거나 기본값 사용
        hidden_dim = getattr(args, 'clf_hidden_dim', 256)
        num_layers = getattr(args, 'clf_num_layers', 2)
        epochs = getattr(args, 'clf_epochs', 20)
        lr = getattr(args, 'clf_lr', 1e-3)
        batch_size = getattr(args, 'clf_batch_size', 128)

        input_dim = X_scaled.shape[1]
        clf_model = MLPClassifier(input_dim, hidden_dim, num_layers).to(device)
        optim_clf = torch.optim.Adam(clf_model.parameters(), lr=lr)
        ce_loss = torch.nn.CrossEntropyLoss()

        dataset_clf = torch.utils.data.TensorDataset(torch.from_numpy(X_scaled).float(), torch.from_numpy(y).long())
        loader_clf = torch.utils.data.DataLoader(dataset_clf, batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            running_loss = 0.0
            for xb, yb in loader_clf:
                xb, yb = xb.to(device), yb.to(device)
                optim_clf.zero_grad()
                out = clf_model(xb)
                loss_clf = ce_loss(out, yb)
                loss_clf.backward()
                optim_clf.step()
                running_loss += loss_clf.item() * yb.size(0)
            if args.verbose:
                print(f"[Task {task_id+1} OOD-MLP] Epoch {ep+1}/{epochs} Loss: {running_loss/len(dataset_clf):.4f}")

        # -----------------------------
        #  저장 (.pt)
        # -----------------------------
        save_path = save_dir / f"clf_{task_id + 1}.pt"
        torch.save({
            "scaler": scaler,
            "state_dict": clf_model.cpu().state_dict(),
            "in_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }, save_path)
        print(f"Task {task_id + 1} 의 OOD MLP 분류기를 {save_path} 에 저장했습니다.")

    def _compute_taskclf_scores(self, model, id_loader, ood_loader, device, args, task_id):
        """저장된 Task-specific 분류기 앙상블을 사용해 ID / OOD 점수를 계산"""
        clf_dir = Path(args.save) / "task_clf"
        if not clf_dir.exists():
            raise ValueError("task_clf 디렉토리가 존재하지 않습니다. 먼저 분류기를 학습하세요.")

        pkl_paths = sorted(clf_dir.glob("clf_*.pt"))
        if task_id is not None and task_id > 0:
            pkl_paths = pkl_paths[:task_id + 1]   # 현재 분류기까지 포함
        else:
            pkl_paths = []

        if not pkl_paths:
            raise ValueError(f"[Task {task_id + 1}] 평가할 이전 OOD 분류기가 없습니다.")

        clfs = []
        for p in pkl_paths:
            d = torch.load(p, map_location='cpu')
            m = MLPClassifier(d['in_dim'], d['hidden_dim'], d['num_layers'])
            m.load_state_dict(d['state_dict'])
            m.eval()
            clfs.append({'scaler': d['scaler'], 'model': m})

        def _score(inputs):
            with torch.no_grad():
                z = model.forward_features(inputs)[:, 0].cpu().numpy()
            scores_list = []
            for d in clfs:
                z_scaled = d['scaler'].transform(z)
                with torch.no_grad():
                    out = d['model'](torch.from_numpy(z_scaled).float())
                    prob_id = torch.softmax(out, dim=1)[:, 0].cpu().numpy()
                scores_list.append(prob_id)
            scores = np.column_stack(scores_list)
            return scores.max(1)

        id_scores, ood_scores = [], []
        model.eval()
        with torch.no_grad():
            for x, _ in tqdm(id_loader, desc="ID"):
                id_scores.append(_score(x.to(device)))
            for x, _ in tqdm(ood_loader, desc="OOD"):
                ood_scores.append(_score(x.to(device)))

        id_scores = np.concatenate(id_scores)
        ood_scores = np.concatenate(ood_scores)
        return torch.from_numpy(id_scores), torch.from_numpy(ood_scores)
    
    def train_one_epoch(self, model: torch.nn.Module, 
                        criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0,
                        set_training_mode=True, task_id=-1, class_mask=None, ema_model=None, args=None):
        model.train(set_training_mode)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop:
                if batch_idx > 20:
                    break
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(input)  # (bs, class + n)
            distill_loss = 0
            if self.distill_head is not None:
                feature = model.forward_features(input)[:, 0]
                output_distill = self.distill_head(feature) 
                mask = torch.isin(torch.tensor(self.labels_in_head), torch.tensor(self.current_classes))
                cur_class_nodes = torch.where(mask)[0]
                m = torch.isin(torch.tensor(self.labels_in_head[cur_class_nodes]), torch.tensor(list(self.added_classes_in_cur_task)))
                distill_node_indices = self.labels_in_head[cur_class_nodes][~m]
                distill_loss = self.kl_div(output[:, distill_node_indices], output_distill[:, distill_node_indices])
               
            if output.shape[-1] > self.num_classes:
                output, _, _ = self.get_max_label_logits(output, class_mask[task_id], slice=False)
                if len(self.added_classes_in_cur_task) > 0:
                    for added_class in self.added_classes_in_cur_task:
                        cur_node = np.where(self.labels_in_head == added_class)[0][-1]
                        output[:, added_class] = output[:, cur_node]
                output = output[:, :self.num_classes]       
                
            if class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, target)
           
            if self.args.CAST:
                if len(self.adapter_vec) > args.k:
                    cur_adapters = model.get_adapter()
                    self.cur_adapters = self.flatten_parameters(cur_adapters)
                    diff_adapter = self.cur_adapters - self.prev_adapters
                    _, other = self.find_same_cluster_items(diff_adapter)
                    sim = 0
                    weights = self.calculate_l2_distance(diff_adapter, other)
                    for o, w in zip(other, weights):
                        if self.args.norm_cast:
                            sim += w * torch.matmul(diff_adapter, o) / (torch.norm(diff_adapter) * torch.norm(o))
                        else:
                            sim += w * torch.matmul(diff_adapter, o)
                    orth_loss = args.beta * torch.abs(sim)
                    if orth_loss > 0:
                        loss += orth_loss
                    
            if self.args.IC:
                if distill_loss > 0:
                    loss += distill_loss
           
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer.step()
            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            if ema_model is not None:
                ema_model.update(model.get_adapter())
            
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def get_max_label_logits(self, output, class_mask, task_id=None, slice=True, target=None):
        for label in range(self.num_classes): 
            label_nodes = np.where(self.labels_in_head == label)[0]
            output[:, label], max_index = torch.max(output[:, label_nodes], dim=1)
        if slice:
            output = output[:, :self.num_classes]
        return output, 0, 0
    
    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader, 
                 device, task_id=-1, class_mask=None, ema_model=None, args=None):
    
        if not self.current_classes or len(self.current_classes) == 0:
            self.current_classes = class_mask[task_id]

        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)
        model.eval()
        correct_sum, total_sum = 0, 0
        label_correct, label_total = np.zeros((self.class_group_size)), np.zeros((self.class_group_size))
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop:
                    if batch_idx > 20:
                        break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(input)
                output, correct, total = self.get_max_label_logits(output, class_mask[task_id], task_id=task_id, target=target, slice=True) 
                output_ema = [output.softmax(dim=1)]
                correct_sum += correct
                total_sum += total
                output = torch.stack(output_ema, dim=-1).max(dim=-1)[0]
                loss = criterion(output, target)
                
                # if self.args.d_threshold and self.current_task + 1 != self.args.num_tasks and self.current_task == task_id:
                #     label_correct, label_total = self.update_acc_per_label(label_correct, label_total, output, target)
                
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            if total_sum > 0:
                print(f"Max Pooling acc: {correct_sum/total_sum}")
                
            if self.args.d_threshold and task_id == self.current_task:
                domain_idx = int(self.label_train_count[self.current_classes][0])
                self.acc_per_label[self.current_classes, domain_idx] += np.round(label_correct / label_total, decimals=3)
                # print(self.label_train_count)
                # print(self.acc_per_label)
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate_till_now(self, model: torch.nn.Module, data_loader, 
                          device, task_id=-1, class_mask=None, acc_matrix=None, ema_model=None, args=None):
        """
        현재까지의 모든 task에 대해 평가하고,
        A_last, A_avg, Forgetting 지표를 계산하여 출력합니다.
        """
        for i in range(task_id + 1):
            test_stats = self.evaluate(model=model, data_loader=data_loader[i]['val'], 
                                       device=device, task_id=i, class_mask=class_mask, ema_model=ema_model, args=args)
            acc_matrix[i, task_id] = test_stats['Acc@1']
        
        A_i = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)
        
        result_str = "[Average accuracy till task{}] A_last: {:.2f} A_avg: {:.2f}".format(task_id+1, A_last, A_avg)


        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += " Forgetting: {:.2f}".format(forgetting)
        else:
            forgetting = 0
        
        if args.wandb:
            import wandb
            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})
        
        print(result_str)
        if args.verbose or args.wandb:
            sub_matrix = acc_matrix[:task_id+1, :task_id+1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            heatmap_path = save_accuracy_heatmap(result, task_id, args)
            if args.wandb:
                import wandb
                wandb.log({"Accuracy Heatmap": wandb.Image(heatmap_path)})
        
        return test_stats

    def flatten_parameters(self, modules):
        flattened_params = []
        for m in modules:
            params = list(m.parameters())
            flattened_params.extend(params)
        return torch.cat([param.view(-1) for param in flattened_params])
    
    def cluster_adapters(self):
        k = self.args.k
        if len(self.adapter_vec) > k:
            self.adapter_vec_array = torch.stack(self.adapter_vec).detach().cpu().numpy().astype(float)
            self.kmeans = KMeans(n_clusters=k, n_init=10)
            self.kmeans.fit(self.adapter_vec_array)
            self.cluster_assignments = self.kmeans.labels_
            print("Cluster(shifts) Assignments:", self.cluster_assignments)
    
    def pre_train_epoch(self, model: torch.nn.Module, epoch: int = 0, task_id: int = 0, args=None):
        if task_id == 0 or args.num_freeze_epochs < 1:
            return model
        if epoch == 0:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = False
            print('Freezing adapter parameters for {} epochs'.format(args.num_freeze_epochs))
        if epoch == args.num_freeze_epochs:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = True
            print('Unfreezing adapter parameters')        
        return model
    
    def pre_train_task(self, model, data_loader, device, task_id, args):
        self.current_task += 1
        self.current_class_group = int(min(self.class_mask[task_id]) / self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]
        print(f"\n\nTASK : {task_id}")
        self.added_classes_in_cur_task = set()  
        if self.class_group_train_count[self.current_class_group] == 0:
            self.distill_head = None
        else:
            if self.args.IC:
                self.distill_head = self.classifier_pool[self.current_class_group]
                inf_acc = self.inference_acc(model, data_loader, device)
                thresholds = []
                if self.args.d_threshold:
                    count = self.class_group_train_count[self.current_class_group]
                    if count > 0:
                        average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                    thresholds = self.args.gamma * (average_accs - inf_acc) / average_accs
                    thresholds = self.tanh(torch.tensor(thresholds)).tolist()
                    thresholds = [round(t, 2) if t > self.args.thre else self.args.thre for t in thresholds]
                    print(f"Thresholds for class {self.current_classes[0]}~{self.current_classes[-1]} : {thresholds}")
                labels_to_be_added = self.detect_labels_to_be_added(inf_acc, thresholds)
                if len(labels_to_be_added) > 0:
                    new_head = self.set_new_head(model, labels_to_be_added, task_id).to(device)
                    model.head = new_head
        optimizer = create_optimizer(args, model)
        with torch.no_grad():
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters)
            self.prev_adapters.requires_grad = False
        if task_id == 0:
            self.task_type_list.append("Initial")
            return model, optimizer
        prev_class = self.class_mask[task_id - 1]
        prev_domain = self.domain_list[task_id - 1]
        cur_class = self.class_mask[task_id]
        self.cur_domain = self.domain_list[task_id]
        if prev_class == cur_class:
            self.task_type = "DIL"
        else:
            self.task_type = "CIL"
        self.task_type_list.append(self.task_type)
        print(f"Current task : {self.task_type}")
        return model, optimizer

    def post_train_task(self, model: torch.nn.Module, task_id=-1):
        self.class_group_train_count[self.current_class_group] += 1
        self.classifier_pool[self.current_class_group] = copy.deepcopy(model.head)
        for c in self.classifier_pool:
            if c is not None:
                for p in c.parameters():
                    p.requires_grad = False
        cur_adapters = model.get_adapter()
        self.cur_adapters = self.flatten_parameters(cur_adapters)
        vector = self.cur_adapters - self.prev_adapters
        self.adapter_vec.append(vector)
        self.adapter_vec_label.append(self.task_type)
        self.cluster_adapters()
                 
    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                           lr_scheduler, device: torch.device, class_mask=None, args=None):
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        max_tasks = getattr(args, 'max_tasks', args.num_tasks)
        ema_model = None
        for task_id in range(args.num_tasks):
            # max_tasks 에 도달하면 반복 중단
            if task_id >= max_tasks:
                print(f"[ICON] Reached max_tasks={max_tasks}. Stopping training loop early.")
                break
            if task_id > 0 and args.reinit_optimizer:
                optimizer = create_optimizer(args, model)
            if task_id == 1 and len(args.adapt_blocks) > 0:
                ema_model = ModelEmaV2(model.get_adapter(), decay=args.ema_decay, device=device)
            model, optimizer = self.pre_train_task(model, data_loader[task_id]['train'], device, task_id, args)
            for epoch in range(args.epochs):
                model = self.pre_train_epoch(model=model, epoch=epoch, task_id=task_id, args=args)
                train_stats = self.train_one_epoch(model=model, criterion=criterion, 
                                                   data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                   device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                   set_training_mode=True, task_id=task_id, class_mask=class_mask, ema_model=ema_model, args=args)
                if lr_scheduler:
                    lr_scheduler.step(epoch)
            self.post_train_task(model, task_id=task_id)
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1 
            test_stats = self.evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, ema_model=ema_model, args=args)
            
            # === Task-specific OOD Classifier 학습 ===
            self.train_task_ood_classifier(model, data_loader, device, args, task_id)
            
            if args.ood_dataset:
                print(f"{'OOD Evaluation':=^60}")
                ood_start = time.time()
                # ID 데이터셋은 모든 태스크의 검증 데이터(ICON의 예측 로직 적용)를 합침
                all_id_datasets = torch.utils.data.ConcatDataset([dl['val'].dataset for dl in data_loader[:task_id+1]])
                # ood_loader는 마지막 태스크에 추가된 ood 데이터 로더 사용
                ood_loader = data_loader[-1]['ood']
                self.evaluate_ood(model, all_id_datasets, ood_loader, device, args, task_id)
                ood_duration = time.time() - ood_start
                print(f"OOD evaluation completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")

            if args.save and utils.is_main_process():
                Path(os.path.join(args.save, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(args.save, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                state_dict = {
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict() if ema_model is not None else None,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    # 추가: head 관련 정보 저장
                    'head_info': {
                        'labels_in_head': self.labels_in_head.tolist(),
                        'added_classes_in_cur_task': list(self.added_classes_in_cur_task),
                        'head_timestamps': self.head_timestamps.tolist(),
                        'num_classes': self.num_classes
                    }
                }
                if args.sched is not None and args.sched != 'constant':
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                utils.save_on_master(state_dict, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,}
            if args.save and utils.is_main_process():
                with open(os.path.join(args.save, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')

    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args, task_id=None):
        model.eval()
        
        vis_task_id = (task_id + 1) 
 
        ood_method = args.ood_method.upper()

        # 1) 데이터셋 크기 맞추기
        id_size, ood_size = len(id_datasets), len(ood_dataset)
        min_size = min(id_size, ood_size)
        if args.develop:
            min_size = 1000
        if args.verbose:
            print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")

        id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
        ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset

        id_loader = torch.utils.data.DataLoader(id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ood_loader = torch.utils.data.DataLoader(ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # 2) 평가할 방법 결정
        if ood_method == "ALL":
            methods = SUPPORTED_METHODS
        else:
            # 쉼표로 구분된 메소드들 처리
            methods = [method.strip().upper() for method in ood_method.split(',')]
            # 지원되지 않는 메소드 확인
            unsupported = [m for m in methods if m not in SUPPORTED_METHODS]
            if unsupported:
                raise ValueError(f"지원되지 않는 OOD 메소드: {unsupported}. 지원되는 메소드: {SUPPORTED_METHODS}")

        from sklearn import metrics
        results = {}

        for method in methods:
            if method == "TASKCLF":
                # 첫 task에서는 TASKCLF가 존재하지 않으므로 MSP 로 대체
                if task_id == 0:
                    print("[TASKCLF] 첫 태스크는 분류기가 없어 MSP로 대체합니다.")
                    actual_method = "MSP"
                    id_scores, ood_scores = compute_ood_scores(actual_method, model, id_loader, ood_loader, device)
                else:
                    id_scores, ood_scores = self._compute_taskclf_scores(model, id_loader, ood_loader, device, args, task_id)
            else:
                id_scores, ood_scores = compute_ood_scores(method, model, id_loader, ood_loader, device)

            # 시각화 및 로깅
            if args.verbose or args.wandb:
                hist_path = save_anomaly_histogram(id_scores.numpy(), ood_scores.numpy(), args, suffix=method.lower(), task_id=vis_task_id)
                if args.wandb:
                    import wandb
                    wandb.log({f"Anomaly Histogram TASK {vis_task_id}": wandb.Image(hist_path)})

            binary_labels = np.concatenate([np.ones(id_scores.shape[0]), np.zeros(ood_scores.shape[0])])
            all_scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])

            fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
            auroc = metrics.auc(fpr, tpr)
            idx_tpr95 = np.abs(tpr - 0.95).argmin()
            fpr_at_tpr95 = fpr[idx_tpr95]

            print(f"[Task {vis_task_id}] [{method}]: AUROC {auroc * 100:.2f}% | FPR@TPR95 {fpr_at_tpr95 * 100:.2f}%")
            if args.wandb:
                import wandb
                wandb.log({f"{method}_AUROC (↑)": auroc * 100, f"{method}_FPR@TPR95 (↓)": fpr_at_tpr95 * 100, "TASK": vis_task_id})

            results[method] = {"auroc": auroc, "fpr_at_tpr95": fpr_at_tpr95, "scores": all_scores}

        return results  # 기존 하드코딩 로직은 도달하지 않음
        # === End new unified OOD evaluation ===

    def restore_head_from_checkpoint(self, model, checkpoint):
        if 'head_info' in checkpoint:
            head_info = checkpoint['head_info']
            self.labels_in_head = np.array(head_info['labels_in_head'])
            self.added_classes_in_cur_task = set(head_info['added_classes_in_cur_task'])
            self.head_timestamps = np.array(head_info['head_timestamps'])
            self.num_classes = head_info['num_classes']
            
            # 확장된 head를 가진 원본 모델과 동일한 크기의 새 head 생성
            orig_head_size = model.head.weight.shape[0]
            target_head_size = len(self.labels_in_head)
            
            if orig_head_size != target_head_size:
                print(f"Adjusting head size from {orig_head_size} to {target_head_size} to match checkpoint")
                in_features = model.head.in_features
                new_head = torch.nn.Linear(in_features, target_head_size)
                model.head = new_head
                
            return model
        else:
            print("Warning: No head information found in checkpoint. Loading model as is.")
            return model
            
    def load_checkpoint(self, model, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f'체크포인트를 찾을 수 없습니다: {checkpoint_path}')
        
        print(f'체크포인트를 로드합니다: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        
        # head 정보 복원
        model = self.restore_head_from_checkpoint(model, checkpoint)
        
        # 모델 파라미터 로드
        model.load_state_dict(checkpoint['model'])

        return model

def init_ICON_default_args(args):
    args.IC = True  #
    args.d_threshold = True  #
    args.gamma = 10.0  
    args.thre = 0 
    args.alpha = 1.0  

    args.CAST = True 
    args.beta = 0.01  
    args.k = 2  
    args.norm_cast = True 

    args.adapt_blocks = [0, 1, 2, 3, 4] 
    args.ema_decay = 0.9999  
    args.num_freeze_epochs = 3  
    args.eval_only_emas = False 

    args.clip_grad = 0.0
    args.reinit_optimizer = True

def init_ICON_CORe50_args(args):

    args.IC = True  #
    args.CAST = True 

    args.d_threshold = True  #

    args.lr = 0.0028125

    args.opt_betas = [0.9, 0.999]
    args.batch_size = 24
    args.ema_decay = 0.9999
    args.k = 3 #Number of Clusters
    args.alpha = 1
    args.beta = 0.05
    args.gamma = 2

    args.adapt_blocks = [0, 1, 2, 3, 4] 
    args.thre = 0 
    args.reinit_optimizer = True
    args.clip_grad = 0.0
    args.num_freeze_epochs = 3  
    args.eval_only_emas = False 
    args.norm_cast = True 

def init_ICON_iDigits_args(args):

    args.IC = True  #
    args.CAST = True 

    args.d_threshold = True  #

    args.lr = 0.0028125

    args.opt_betas = [0.9, 0.999]
    args.batch_size = 24
    args.ema_decay = 0.9999
    args.k = 2 #Number of Clusters
    args.alpha = 1
    args.beta = 0.05
    args.gamma = 2

    args.adapt_blocks = [0, 1, 2, 3, 4] 
    args.thre = 0.5
    args.reinit_optimizer = True
    args.clip_grad = 0.0
    args.num_freeze_epochs = 3  
    args.eval_only_emas = False 
    args.norm_cast = True 

def init_ICON_DomainNet_args(args):

    args.IC = True  #
    args.CAST = True 

    args.d_threshold = True  #

    args.lr = 0.0028125

    args.opt_betas = [0.9, 0.999]
    args.batch_size = 24
    args.ema_decay = 0.9999
    args.k = 3 #Number of Clusters
    args.alpha = 1
    args.beta = 0.01
    args.gamma = 2

    args.adapt_blocks = [0, 1, 2, 3, 4] 
    args.thre = 0 
    args.reinit_optimizer = True
    args.clip_grad = 0.0
    args.num_freeze_epochs = 3  
    args.eval_only_emas = False 
    args.norm_cast = True 



