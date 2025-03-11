import argparse
import os
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from utils import seed_everything
from sklearn.metrics import confusion_matrix

import continual_datasets.continual_datasets as cd
from continual_datasets.dataset_utils import UnknownWrapper, RandomSampleWrapper

class OVANet(nn.Module):
    def __init__(self, num_known=10, model_name='vit_base_patch16_224'):
        super(OVANet, self).__init__()
        self.num_known = num_known
        self.vit = timm.create_model(model_name, pretrained=True)
        #분류기 제거
        if hasattr(self.vit, 'head'):
            in_features = self.vit.head.in_features
            self.vit.reset_classifier(0)

        for p in self.vit.parameters():
            p.requires_grad = False
        self.in_dim = in_features

        # Closed-set classifier (C1)
        self.closed_fc = nn.Linear(self.in_dim, num_known)

        # Open-set classifier (C2)
        self.open_fc = nn.Linear(self.in_dim, num_known * 2)

    def forward(self, x):
        """
        x: [B, C, H, W]
        반환:
         feat         : [B, in_dim]
         closed_logits: [B, num_known]
         open_logits  : [B, num_known, 2] (각 known 클래스에 대해 inlier/unknown 점수)
        """
        feat = self.vit.forward_features(x)[:, 0]
        closed_logits = self.closed_fc(feat)
        open_logits = self.open_fc(feat)
        open_logits = open_logits.view(-1, self.num_known, 2)
        return feat, closed_logits, open_logits

def compute_ova_loss(out_open, labels, num_known):
    """
    out_open: [B, num_known, 2] → 먼저 transpose하여 [B, 2, num_known]
    labels: [B] (정답 클래스: 0~num_known-1)
    
    OVA loss:
      For each sample, 
        positive loss = -log( p_inlier(correct_class) )
        negative loss = max_{c != true} -log( 1 - p_inlier(c) )
    """
    out_open = out_open.transpose(1, 2)  # [B, 2, num_known]
    p_open = F.softmax(out_open, dim=1)   # [B, 2, num_known]
    B = labels.size(0)
    one_hot = torch.zeros(B, num_known, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    pos_loss = -torch.log(p_open[:, 1, :] + 1e-8)
    pos_loss = (pos_loss * one_hot).sum(dim=1)
    neg_loss = -torch.log(p_open[:, 0, :] + 1e-8)
    neg_loss = neg_loss * (1 - one_hot)
    neg_loss, _ = neg_loss.max(dim=1)
    return pos_loss.mean(), neg_loss.mean()

def train_ovanet(model, train_loader, optimizer, device, epochs=3, log_interval=50, num_known=10):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_ce, total_ova, total_loss = 0.0, 0.0, 0.0
        step_count = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            _, out_closed, out_open = model(images)
            loss_ce = ce_loss_fn(out_closed, labels)
            pos_loss, neg_loss = compute_ova_loss(out_open, labels, num_known)
            ova_loss = 0.5 * (pos_loss + neg_loss)
            loss = loss_ce + ova_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_ce += loss_ce.item()
            total_ova += ova_loss.item()
            total_loss += loss.item()
            step_count += 1

            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"CE Loss: {total_ce/step_count:.4f} OVA Loss: {total_ova/step_count:.4f} Total Loss: {total_loss/step_count:.4f}")
    return model

def test_ovanet(model, test_loader, device, threshold=0.5):
    """
        - Closed-set classifier로 후보 클래스를 결정한 후,
      - 해당 클래스의 open-set inlier 확률(softmax)을 확인하여, 
        p_inlier < threshold이면 unknown (레이블 = num_known)으로 예측.
    반환: y_true, y_pred (list[int])
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, out_closed, out_open = model(images)
            # 후보: argmax from closed-set classifier
            cand = out_closed.argmax(dim=1)  # [B]
            # open-set: reshape to [B, 2, num_known] and softmax
            out_open = out_open.transpose(1, 2)
            p_open = F.softmax(out_open, dim=1)  # [B, 2, num_known]
            B = images.size(0)
            for i in range(B):
                c = cand[i].item()
                p_inlier = p_open[i, 1, c].item()  # inlier 확률 for candidate
                p_inlier = p_open[i, 1, c].item()
                if p_inlier < threshold:
                    y_pred.append(model.num_known)  # unknown
                else:
                    y_pred.append(c)
            y_true.extend(labels.cpu().numpy().tolist())
    return y_true, y_pred

def compute_hscore(y_true, y_pred, num_known):
    """
    H-Score = 2 * (acc_known * acc_unknown) / (acc_known + acc_unknown)
      - known: label 0 ~ num_known-1
      - unknown: label == num_known
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    known_mask = (y_true < num_known)
    unknown_mask = (y_true == num_known)
    if known_mask.sum() == 0 or unknown_mask.sum() == 0:
        return 0.0, 0.0, 0.0
    acc_known = np.mean(y_true[known_mask] == y_pred[known_mask])
    acc_unknown = np.mean(y_true[unknown_mask] == y_pred[unknown_mask])
    if acc_known + acc_unknown == 0:
        return 0.0, acc_known, acc_unknown
    h = 2 * (acc_known * acc_unknown) / (acc_known + acc_unknown)
    return h, acc_known, acc_unknown

def get_dataset(dataset_class_name, train, data_path, transform, **kwargs):
    if hasattr(cd, dataset_class_name):
        dataset_cls = getattr(cd, dataset_class_name)
    else:
        raise ValueError(f"Dataset class {dataset_class_name} not found in continual_datasets.")
    return dataset_cls(data_path, train=train, download=True, transform=transform, **kwargs)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/local_datasets')
    parser.add_argument('--train_dataset', type=str, default='MNIST_RGB', help='학습에 사용할 데이터셋 클래스 이름 (예: MNIST_RGB)')
    parser.add_argument('--test_known_dataset', type=str, default='MNIST_RGB', help='테스트용 known 데이터셋 클래스 이름 (예: MNIST_RGB)')
    parser.add_argument('--test_unknown_dataset', type=str, default='EMNIST_RGB', help='테스트용 unknown 데이터셋 클래스 이름 (예: EMNIST_RGB)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--num_known', type=int, default=10, help='known 클래스 수 (예: mnist 10)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = get_dataset(args.train_dataset, train=True, data_path=args.data_path, transform=train_transform)
    test_known = get_dataset(args.test_known_dataset, train=False, data_path=args.data_path, transform=test_transform)
    test_unknown = get_dataset(args.test_unknown_dataset, train=False, data_path=args.data_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # unknown dataset의 라벨을 모두 num_known 값으로 변경
    test_unknown = UnknownWrapper(test_unknown, unknown_label=args.num_known)
    
    # 두 테스트 데이터셋의 샘플 수를 랜덤하게 맞추기 위해 RandomSampleWrapper 사용
    n_samples = min(len(test_known), len(test_unknown))
    test_known_subset = RandomSampleWrapper(test_known, num_samples=n_samples, seed=args.seed)
    test_unknown_subset = RandomSampleWrapper(test_unknown, num_samples=n_samples, seed=args.seed + 1)
    
    # 두 데이터셋을 합쳐서 test dataset 구성
    test_dataset = ConcatDataset([test_known_subset, test_unknown_subset])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = OVANet(num_known=args.num_known, model_name=args.model_name)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print("===== Training OVANet =====")
    model = train_ovanet(model, train_loader, optimizer, device, epochs=args.epochs, num_known=args.num_known)

    print("===== Testing OVANet =====")
    y_true, y_pred = test_ovanet(model, test_loader, device, threshold=args.threshold)
    h_score, acc_known, acc_unknown = compute_hscore(y_true, y_pred, args.num_known)
    print(f"H-score: {h_score:.4f} | Known Accuracy: {acc_known:.4f} | Unknown Accuracy: {acc_unknown:.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(args.num_known + 1)))
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    main()
