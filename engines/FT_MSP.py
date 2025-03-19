import os
import sys
import math
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.utils import accuracy
import utils
from sklearn.metrics import roc_auc_score

class Engine:
    def __init__(self, model=None, device=None, class_mask=[], domain_list=[], args=None):
        """
        Args:
            model: incremental learning에서 fine‑tuning할 모델 (예: ViT).
            device: torch.device 객체.
            class_mask, domain_list: incremental dataset 관련 정보 (build_incremental_scenario.py에서 전달됨).
            args: argparse 인자 (num_tasks, epochs, batch_size, lr, output_dir, unknown_class, threshold 등 포함).
        """
        self.model = model
        self.device = device
        self.args = args
        self.class_mask = class_mask
        self.domain_list = domain_list
        self.num_tasks = args.num_tasks

    def train_one_epoch(self, model, criterion, data_loader, optimizer, device, epoch, args):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Train: Epoch [{epoch+1}/{args.epochs}]"
        for batch_idx, (inputs, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop:
                if batch_idx>20:
                    break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)  # 모델의 forward는 logits를 반환한다고 가정
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            metric_logger.update(Loss=loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inputs.size(0))
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inputs.size(0))
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def evaluate_task(self, model, data_loader, device, args):
        """
        ID 데이터에 대해 평가합니다.
        """
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop:
                    if batch_idx>20:
                        break
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                metric_logger.update(Loss=loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=inputs.size(0))
                metric_logger.meters['Acc@5'].update(acc5.item(), n=inputs.size(0))
        print("* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} Loss {losses.global_avg:.3f}".format(
            top1=metric_logger.meters['Acc@1'],
            top5=metric_logger.meters['Acc@5'],
            losses=metric_logger.meters['Loss']
        ))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def evaluate_ood(self, model, id_loader, ood_loader, device, args):
        """
        OOD 평가: 모델 출력 logits에서 softmax의 최대값을 id_score로,
        ood_score = 1 - id_score로 정의한 후, ID와 OOD 데이터를 구분하는 성능(예: AUROC 등)을 계산합니다.
        
        Args:
            id_loader: incremental 학습 시의 ID validation 데이터 로더.
            ood_loader: OOD 평가용 데이터 로더 (예: MNIST, 모든 라벨이 unknown_class로 변환된).
            args.unknown_class: OOD 데이터의 라벨 (예: 10).
            args.threshold: OOD detection 임계값.
        """
        # ID 데이터 평가
        model.eval()
        all_id_logits = []
        all_id_targets = []
        with torch.no_grad():
            for inputs, targets in id_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                all_id_logits.append(outputs)
                all_id_targets.append(targets.cpu())
        all_id_logits = torch.cat(all_id_logits, dim=0)
        all_id_targets = torch.cat(all_id_targets, dim=0)
        id_softmax = F.softmax(all_id_logits, dim=1)
        max_softmax, _ = torch.max(id_softmax, dim=1)
        id_scores = max_softmax  # 높은 값이면 ID 데이터임
        id_preds = all_id_logits.argmax(dim=1)
        id_acc = accuracy(all_id_logits, all_id_targets, topk=(1,))[0].item()

        # OOD 데이터 평가
        all_ood_logits = []
        all_ood_targets = []
        with torch.no_grad():
            for inputs, targets in ood_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                all_ood_logits.append(outputs)
                all_ood_targets.append(targets.cpu())
        all_ood_logits = torch.cat(all_ood_logits, dim=0)
        all_ood_targets = torch.cat(all_ood_targets, dim=0)
        ood_softmax = F.softmax(all_ood_logits, dim=1)
        max_softmax_ood, _ = torch.max(ood_softmax, dim=1)
        # OOD score: 낮은 max softmax이면 OOD로 판단하기 위해 1 - max_softmax 사용
        ood_scores = 1 - max_softmax_ood

        # OOD 예측: ood_score가 threshold 이상이면 OOD로 예측 (즉, label = args.unknown_class)
        ood_preds = (ood_scores >= args.threshold).long()
        # 실제 OOD 데이터의 모든 target은 args.unknown_class로 설정되어 있으므로, ood detection accuracy는
        # ood_preds가 1인 비율
        ood_acc = np.mean(ood_preds.cpu().numpy() == 1)

        # AUROC 계산: ID는 positive (1), OOD는 negative (0)
        id_binary = np.ones(all_id_logits.shape[0], dtype=np.int32)
        ood_binary = np.zeros(all_ood_logits.shape[0], dtype=np.int32)
        binary_labels = np.concatenate([id_binary, ood_binary])
        combined_scores = torch.cat([max_softmax, max_softmax_ood]).cpu().numpy()
        try:
            auc_roc = roc_auc_score(binary_labels, combined_scores)
        except Exception:
            auc_roc = 0.0

        print(f"OOD Evaluation: ID Acc: {id_acc:.3f}, OOD Acc: {ood_acc:.3f}, AUROC: {auc_roc:.3f}")
        return id_acc, ood_acc, auc_roc

    def train_and_evaluate(self, model, criterion, data_loader, optimizer, lr_scheduler, device, args, ood_loader=None):
        """
        incremental learning dataset의 각 task에 대해 순차적으로 fine‑tuning을 진행합니다.
        data_loader: 각 task별 {'train': DataLoader, 'val': DataLoader}로 구성된 리스트.
        ood_loader: OOD 평가용 DataLoader (예: MNIST, UnknownWrapper 적용), 선택 사항.
        """
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            print(f"\n--- Training on Task {task_id+1}/{args.num_tasks} ---")
            for epoch in range(args.epochs):
                train_stats = self.train_one_epoch(model, criterion, data_loader[task_id]['train'], optimizer, device, epoch, args)
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch)
            # 각 task 학습 후, 해당 task의 validation 데이터에 대해 평가
            print(f"\n=== Task {task_id+1} Evaluation ===")
            stats = self.evaluate_task(model, data_loader[task_id]['val'], device, args)
            acc_matrix[task_id, task_id] = stats['Acc@1']
            print(f"Task {task_id+1} evaluation: Acc@1 = {stats['Acc@1']:.2f}")
        # 전체 incremental 학습 완료 후, OOD 평가 (ood_loader가 제공된 경우)
        if ood_loader is not None:
            print("\n=== OOD Evaluation ===")
            # ID 데이터: 모든 task의 validation 데이터를 합침
            all_id_datasets = [dl['val'].dataset for dl in data_loader]
            combined_id_dataset = torch.utils.data.ConcatDataset(all_id_datasets)
            combined_id_loader = torch.utils.data.DataLoader(combined_id_dataset,
                                                              batch_size=args.batch_size,
                                                              shuffle=False,
                                                              num_workers=args.num_workers,
                                                              pin_memory=args.pin_mem)
            id_acc, ood_acc, auc_roc = self.evaluate_ood(model, combined_id_loader, ood_loader, device, args)
            print(f"Overall: ID Acc: {id_acc:.2f}, OOD Acc: {ood_acc:.2f}, AUROC: {auc_roc:.2f}")
        else:
            raise ValueError("OOD loader가 제공되지 않았습니다.")
        return stats

    def save_checkpoint(self, model, optimizer, epoch, args):
        checkpoint_dir = os.path.join(args.output_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_task_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args)
        }, checkpoint_path)
        print(f"Checkpoint 저장: {checkpoint_path}")

    def load_checkpoint(self, model, optimizer, checkpoint_path, device):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Checkpoint {checkpoint_path} 불러옴 (Epoch {checkpoint['epoch']})")
            return checkpoint["epoch"]
        else:
            print(f"Checkpoint {checkpoint_path} 없음")
            return 0

# if __name__ == "__main__":
#     import argparse
#     from timm import create_model
#     from torchvision import datasets, transforms
#     # 간단한 예시: CIFAR10을 incremental ID 데이터셋, MNIST를 OOD 데이터셋으로 사용
#     parser = argparse.ArgumentParser("FT_MSP_ENGINE: Finetuning ViT with MSP-based OOD Detection for Incremental Learning")
#     parser.add_argument("--batch-size", default=24, type=int, help="배치 사이즈")
#     parser.add_argument("--epochs", default=5, type=int, help="각 task 당 에폭 수")
#     parser.add_argument("--lr", default=0.001, type=float, help="학습률")
#     parser.add_argument("--output-dir", default="./output", help="체크포인트 저장 경로")
#     parser.add_argument("--print-freq", default=100, type=int, help="출력 빈도")
#     parser.add_argument("--device", default="cuda", help="학습/평가 디바이스")
#     parser.add_argument("--num-workers", default=4, type=int, help="DataLoader num_workers")
#     # 모델 관련 인자
#     parser.add_argument("--model", default="vit_base_patch16_224", type=str, help="학습할 모델 이름")
#     parser.add_argument("--pretrained", default=True, type=bool, help="Pretrained 모델 로드 여부")
#     parser.add_argument("--nb-classes", default=10, type=int, help="클래스 개수")
#     # OOD 평가 관련 인자
#     parser.add_argument("--unknown-class", default=10, type=int, help="OOD 데이터의 라벨 (예: 10)")
#     parser.add_argument("--threshold", default=0.5, type=float, help="OOD score 임계값")
#     # Incremental learning 관련 인자
#     parser.add_argument("--num-tasks", default=5, type=int, help="Incremental tasks 수")
#     args = parser.parse_args()

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     model = create_model(args.model, pretrained=args.pretrained, num_classes=args.nb_classes)
#     # FT fine-tuning: head만 업데이트
#     for n, p in model.named_parameters():
#         p.requires_grad = False
#         if "head" in n:
#             p.requires_grad = True
#     model.to(device)
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
#     lr_scheduler = None

#     # 예시 incremental dataset: CIFAR10을 5개 task로 분할 (각 task 당 2개 클래스)
#     transform = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#     full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
#     full_val = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
#     tasks = []
#     num_classes_per_task = args.nb_classes // args.num_tasks  # 예: 10 // 5 = 2
#     for t in range(args.num_tasks):
#         train_indices = [i for i, (_, label) in enumerate(full_train) if label // num_classes_per_task == t]
#         val_indices = [i for i, (_, label) in enumerate(full_val) if label // num_classes_per_task == t]
#         train_subset = torch.utils.data.Subset(full_train, train_indices)
#         val_subset = torch.utils.data.Subset(full_val, val_indices)
#         tasks.append({
#             "train": torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True),
#             "val": torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#         })

#     # OOD 데이터셋: MNIST를 사용하고, 모든 라벨을 unknown_class(args.unknown_class)로 변경
#     mnist_transform = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#     ])
#     from continual_datasets.dataset_utils import UnknownWrapper
#     ood_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)
#     ood_dataset = UnknownWrapper(ood_dataset, unknown_label=args.unknown_class)
#     ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

#     criterion = nn.CrossEntropyLoss().to(device)
#     engine = Engine(model=model, device=device, args=args)
#     engine.train_and_evaluate(model, criterion, tasks, optimizer, lr_scheduler, device, args, ood_loader=ood_loader)
#     engine.save_checkpoint(model, optimizer, args.epochs - 1, args)
