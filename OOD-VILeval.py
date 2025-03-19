import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from timm.models import create_model
from sklearn.metrics import roc_auc_score
from timm.utils import accuracy

from continual_datasets.base_datasets import MNIST_RGB, MNISTM, SynDigit, CORe50, DomainNet, EMNIST_RGB
from continual_datasets.dataset_utils import UnknownWrapper, RandomSampleWrapper

def load_dataset(name, data_path, transform = None, is_id=True, unknown_label=None):
    """
    데이터셋 로드 함수:
      - is_id=True: 원래 라벨 사용 (ID 데이터셋)
      - is_id=False: UnknownWrapper를 통해 모든 라벨을 unknown_label로 변경 (OOD 데이터셋)
    """
    name = name.lower()
    if name == "mnist":
        dataset = MNIST_RGB(root=data_path, train=False, transform=transform, download=True)
    elif name == "mnistm":
        dataset = MNISTM(root=data_path, train=False, transform=transform, download=True)
    elif name == "synthdigit":
        dataset = SynDigit(root=data_path, train=False, transform=transform, download=True)
    elif name == "core50":
        dataset = CORe50(root=data_path, train=False, transform=transform, download=True)
    elif name == "domainnet":
        dataset = DomainNet(root=data_path, train=False, transform=transform, download=True)
    elif name == "emnist":
        dataset = EMNIST_RGB(root=data_path, split='letters', train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    if not is_id:
        if unknown_label is None:
            raise ValueError("unknown_label must be provided for OOD dataset")
        dataset = UnknownWrapper(dataset, unknown_label)
    return dataset


def evaluate_ood(model, dataloader, device, unknown_class):
    """
    평가 함수: 모델의 예측 결과를 바탕으로
      - ID Accuracy, OOD Accuracy, H-score 및 AUC-ROC를 계산합니다.
    모델의 forward 함수는 (logits, ood_score)를 반환합니다.
    """
    model.eval()
    all_logits, all_ood_scores, all_targets = [], [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits, ood_score = model(inputs)  # logits: [B, num_classes], ood_score: [B]
            all_logits.append(logits)
            all_ood_scores.append(ood_score)
            all_targets.append(targets.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_ood_scores = torch.cat(all_ood_scores, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    y_true = all_targets.numpy()
    
    id_mask = y_true < unknown_class
    ood_mask = y_true == unknown_class
    

    id_logits = all_logits[id_mask]
    id_targets = all_targets[id_mask].to(device)
    id_acc = accuracy(id_logits, id_targets, topk=(1,))[0].item()

    
    # OOD Accuracy
    # ood_score가 0.5 이상이면 OOD로 예측
    threshold = 0.5
    # ood_preds: 1이면 ID, 0이면 OOD로 예측 (ood_score가 낮으면 ID, 높으면 OOD)
    ood_preds = (all_ood_scores < threshold).cpu().numpy().astype(np.int32)  # ID:1, OOD:0
    if ood_mask.sum() > 0:
        ood_acc = np.mean(ood_preds[ood_mask] == 0)
    else:
        raise ValueError("OOD 데이터가 없습니다.")

    # H-score: ID와 OOD Accuracy의 조화평균
    h_score = 2 * (id_acc * ood_acc) / (id_acc + ood_acc) if (id_acc + ood_acc) > 0 else 0.0

    # AUROC: 이진 분류 (ID: 1, OOD: 0)로 계산
    y_binary = (y_true < unknown_class).astype(np.int32)
    try:
        auc_roc = roc_auc_score(y_binary, all_ood_scores.cpu().numpy())
    except ValueError:
        auc_roc = 0.0

    return id_acc, ood_acc, h_score, auc_roc

